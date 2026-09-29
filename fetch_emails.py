import base64
import json

from gmail_auth import get_gmail_service

MAX_RESULTS = 100
OUTPUT_FILE = "emails.json"


def get_header(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def get_body(payload):
    if payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        text = get_body(part)
        if text:
            return text

    return ""


def fetch_latest_emails(service, max_results=MAX_RESULTS):
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    emails = []
    for msg_ref in messages:
        msg = service.users().messages().get(userId="me", id=msg_ref["id"], format="full").execute()
        headers = msg["payload"].get("headers", [])
        emails.append(
            {
                "id": msg["id"],
                "threadId": msg["threadId"],
                "from": get_header(headers, "From"),
                "to": get_header(headers, "To"),
                "subject": get_header(headers, "Subject"),
                "date": get_header(headers, "Date"),
                "snippet": msg.get("snippet", ""),
                "body": get_body(msg["payload"]),
            }
        )

    return emails


if __name__ == "__main__":
    service = get_gmail_service()
    emails = fetch_latest_emails(service)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)
    print(f"Fetched {len(emails)} emails, saved to {OUTPUT_FILE}")
