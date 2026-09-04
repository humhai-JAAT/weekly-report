"""Runs the weekly Nifty 500 buy-signal scan (weekly_buy_scan.py), renders the
"genuinely fresh" list as a PDF, and sends it to one WhatsApp number via the
Meta WhatsApp Business Cloud API. Built for the scheduled GitHub Actions job
(.github/workflows/weekly_buy_signal_report.yml) that runs every Friday
18:00 IST, after the weekly candle has closed.

Only sends the "fresh" list (genuinely first-fire-this-arm-cycle stocks) —
NOT the "repeat within arm cycle" ones, since those aren't real trade
opportunities (see weekly_buy_scan.split_hits's docstring). If nothing is
fresh this week, still sends a one-line "no signals this week" message
(via the same template's body text) rather than silently sending nothing.

Requires 4 secrets: WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID,
WHATSAPP_RECIPIENT_NUMBER, WHATSAPP_TEMPLATE_NAME. The template (approved
2026-09-01, name "weekly_buy_signal_report") has a DOCUMENT header (dynamic
media) and a BODY with 2 NAMED placeholders: {{stock_count}} and
{{report_date}} — Meta requires lowercase/underscore named variables now,
not the older {{1}}/{{2}} positional style, so the API call below sends each
body parameter with a `parameter_name` key matching the template exactly.
See send_whatsapp_document() for the exact payload shape.
"""

import io
import os

import pandas as pd
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from weekly_buy_scan import scan, split_hits

GRAPH_API_VERSION = "v20.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


def build_pdf(fresh: pd.DataFrame, scan_date: str) -> bytes:
    """Renders the fresh-signal list as a one-page PDF, in-memory (no temp
    file — the bytes go straight into the WhatsApp media upload below)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Weekly Buy Signal Report — {scan_date}", styles["Title"]),
        Paragraph(f"Nifty 500 universe · EMA-MACD V2.1.2 strategy · weekly timeframe · "
                  f"{len(fresh)} genuinely fresh signal(s)", styles["Normal"]),
        Spacer(1, 0.5 * cm),
    ]

    if fresh.empty:
        story.append(Paragraph("No genuinely fresh buy signals this week.", styles["Normal"]))
    else:
        header = ["Symbol", "Company", "Price @ Signal", "Signal Date", "LTP", "% Change"]
        data = [header]
        for _, row in fresh.iterrows():
            data.append([
                row["symbol"], row["company"][:35],
                f"{row['price_at_signal']:.2f}", str(row["buy_signal_date"]),
                f"{row['ltp']:.2f}", f"{row['pct_change']:+.2f}%",
            ])
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Reproduces the pre-2026-08-13 arm-cycle staleness bug intentionally (research mode) — "
        "see strategy.py's MAX_ARM_CYCLE_AGE_DAYS comment.",
        styles["Italic"],
    ))
    doc.build(story)
    return buf.getvalue()


def _raise_with_meta_detail(resp: requests.Response) -> None:
    """`resp.raise_for_status()` alone only gives a bare '401 Client Error:
    Unauthorized' with no explanation — Meta's own error body (which has a
    real message/type/code explaining WHY, e.g. an expired token vs a
    permission the System User doesn't have) gets silently discarded.
    Live-hit 2026-09-04: two straight 401s with zero detail made it
    impossible to tell whether the problem was the token itself or
    something else (app assignment, phone number mismatch) without this."""
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        raise requests.exceptions.HTTPError(f"{e} | body: {detail}", response=resp) from None


def upload_media(token: str, phone_number_id: str, pdf_bytes: bytes, filename: str) -> str:
    """Uploads the PDF to Meta's Media API, returns the media_id used to
    reference it in the template send call below — avoids needing a
    separate public file host."""
    resp = requests.post(
        f"{GRAPH_API_BASE}/{phone_number_id}/media",
        headers={"Authorization": f"Bearer {token}"},
        data={"messaging_product": "whatsapp", "type": "application/pdf"},
        files={"file": (filename, pdf_bytes, "application/pdf")},
        timeout=30,
    )
    _raise_with_meta_detail(resp)
    return resp.json()["id"]


def send_whatsapp_document(token: str, phone_number_id: str, recipient: str, template_name: str,
                            media_id: str, filename: str, stock_count: int, scan_date: str) -> dict:
    """Sends the approved DOCUMENT-header template with the uploaded PDF as
    its dynamic media, and 2 body text params (count, date). Must be a
    template (not a free-form message) since this runs unprompted, outside
    any 24h customer-service window."""
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": "en_US"},
            "components": [
                {
                    "type": "header",
                    "parameters": [{"type": "document", "document": {"id": media_id, "filename": filename}}],
                },
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "parameter_name": "stock_count", "text": str(stock_count)},
                        {"type": "text", "parameter_name": "report_date", "text": scan_date},
                    ],
                },
            ],
        },
    }
    resp = requests.post(
        f"{GRAPH_API_BASE}/{phone_number_id}/messages",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload, timeout=30,
    )
    _raise_with_meta_detail(resp)
    return resp.json()


def main() -> None:
    # .strip() on every secret: GitHub Actions' own env dump showed a blank
    # line right after WHATSAPP_TOKEN's redacted value, meaning the secret
    # itself carries an embedded newline no matter how it was copied
    # (Meta's own token-copy UI appears to append one) - this caused
    # `requests` to reject the Authorization header as malformed
    # (InvalidHeader) before the request even left the machine. Stripping
    # here makes the script robust to that regardless of how any of these
    # 4 secrets get pasted in the future.
    token = os.environ["WHATSAPP_TOKEN"].strip()
    phone_number_id = os.environ["WHATSAPP_PHONE_NUMBER_ID"].strip()
    recipient = os.environ["WHATSAPP_RECIPIENT_NUMBER"].strip()
    template_name = os.environ["WHATSAPP_TEMPLATE_NAME"].strip()

    result = scan()
    fresh, _repeats = split_hits(result)
    scan_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    pdf_bytes = build_pdf(fresh, scan_date)
    filename = f"weekly_buy_signals_{scan_date}.pdf"

    media_id = upload_media(token, phone_number_id, pdf_bytes, filename)
    print(f"Uploaded PDF, media_id={media_id}")

    response = send_whatsapp_document(token, phone_number_id, recipient, template_name,
                                       media_id, filename, len(fresh), scan_date)
    print(f"WhatsApp send response: {response}")


if __name__ == "__main__":
    main()
