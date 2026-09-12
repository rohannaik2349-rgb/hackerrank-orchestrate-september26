"""
HackerRank Orchestrate - Financial Affordability Agent
Place this file at: code/main.py
Run with: python3 code/main.py
"""

import os
import csv
import json
import base64
import re
from datetime import date, timedelta, datetime
from collections import defaultdict
from typing import Optional

# ── Anthropic client ──────────────────────────────────────────────────────────
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    USE_AI = True
except Exception:
    USE_AI = False
    print("[WARN] anthropic package not installed or no API key. AI features disabled.")

MODEL = "claude-sonnet-4-6"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET = "dataset"
OUTPUT_PATH = "output.csv"

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filename: str) -> list[dict]:
    path = os.path.join(DATASET, filename)
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_all_data():
    return {
        "requests":        load_csv("requests.csv"),
        "profiles":        load_csv("financial_profiles.csv"),
        "events":          load_csv("financial_events.csv"),
        "payment_options": load_csv("request_payment_options.csv"),
        "exchange_rates":  load_csv("exchange_rates.csv"),
        "messages":        load_csv("messages.csv"),
        "images":          load_csv("images.csv"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CURRENCY CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def build_rate_lookup(exchange_rates: list[dict]) -> dict:
    """
    Returns: { (date_str, from_currency, to_currency): rate }
    """
    lookup = {}
    for row in exchange_rates:
        key = (row["date"], row["from_currency"], row["to_currency"])
        lookup[key] = float(row["rate"])
    return lookup


def convert(amount: float, from_cur: str, to_cur: str,
            rate_lookup: dict, on_date: str) -> float:
    if from_cur == to_cur:
        return amount
    key = (on_date, from_cur, to_cur)
    if key in rate_lookup:
        return amount * rate_lookup[key]
    # Try reverse
    rev = (on_date, to_cur, from_cur)
    if rev in rate_lookup:
        return amount / rate_lookup[rev]
    return amount  # fallback: assume 1:1


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE / AMOUNT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_amount_from_image(image_row: dict) -> Optional[float]:
    """
    Use Claude vision to pull a numeric amount from a payslip / receipt image.
    Falls back to None if AI is unavailable.
    """
    if not USE_AI:
        return None

    image_path = os.path.join(DATASET, "media", "images", image_row.get("filename", ""))
    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    media_type = media_map.get(ext, "image/jpeg")

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": media_type,
                                                  "data": b64}},
                    {"type": "text",
                     "text": ("Extract the primary monetary amount from this document. "
                              "Reply with ONLY a plain number (no currency symbol, no commas). "
                              "Example: 45000.00")}
                ]
            }]
        )
        text = resp.content[0].text.strip()
        return float(re.sub(r"[^\d.]", "", text))
    except Exception as e:
        print(f"[WARN] Image extraction failed: {e}")
        return None


def resolve_blank_amounts(events: list[dict], images: list[dict]) -> list[dict]:
    """
    For events with blank amounts, look up linked images and extract amounts.
    """
    # Build image lookup: related_event_id -> image row
    img_lookup = {row["related_event_id"]: row for row in images
                  if row.get("related_event_id")}

    for ev in events:
        if ev.get("amount", "").strip() == "":
            img = img_lookup.get(ev["event_id"])
            if img:
                extracted = extract_amount_from_image(img)
                if extracted is not None:
                    ev["amount"] = str(extracted)
                    print(f"[INFO] Extracted amount {extracted} for event {ev['event_id']}")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# 4. FINANCIAL STATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(s: str) -> Optional[date]:
    if not s or not s.strip():
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def safe_float(s) -> float:
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def build_user_state(user_id: str, request_date: date, data: dict,
                     rate_lookup: dict, home_currency: str) -> dict:
    """
    Returns a dict with:
      - balance: current balance on request_date
      - min_balance: floor to always stay above
      - recurring: list of {event_id, amount, frequency, next_date, is_flexible, is_essential}
      - confirmed_income: list of {event_id, amount, date}
      - pending: list of {event_id, amount, date}  (one-time future debits)
      - profile: the full profile row
    """
    # Profile
    profile = next((p for p in data["profiles"] if p["user_id"] == user_id), {})
    balance = safe_float(profile.get("balance", 0))
    min_balance = safe_float(profile.get("minimum_balance", 0))

    events = [e for e in data["events"] if e["user_id"] == user_id]

    recurring = []
    confirmed_income = []
    pending_debits = []
    seen_event_ids = set()

    for ev in events:
        eid = ev["event_id"]
        # De-duplicate
        if eid in seen_event_ids:
            continue
        seen_event_ids.add(eid)

        ev_date = parse_date(ev.get("date", ""))
        amount = safe_float(ev.get("amount", 0))
        ev_type = ev.get("type", "").lower()
        status = ev.get("status", "").lower()
        is_recurring = ev.get("is_recurring", "false").lower() == "true"
        is_flexible = ev.get("is_flexible", "false").lower() == "true"
        is_essential = ev.get("is_essential", "true").lower() == "true"
        currency = ev.get("currency", home_currency)
        frequency = ev.get("frequency", "monthly")

        # Convert to home currency
        if ev_date:
            amount = convert(amount, currency, home_currency,
                             rate_lookup, ev_date.isoformat())
        else:
            amount = convert(amount, currency, home_currency,
                             rate_lookup, request_date.isoformat())

        if is_recurring:
            recurring.append({
                "event_id": eid,
                "amount": amount,
                "frequency": frequency,
                "next_date": ev_date,
                "is_flexible": is_flexible,
                "is_essential": is_essential,
                "name": ev.get("name", eid),
                "type": ev_type,
            })
        elif ev_type in ("income", "salary", "credit") and status == "confirmed":
            if ev_date and ev_date > request_date:
                confirmed_income.append({
                    "event_id": eid,
                    "amount": amount,
                    "date": ev_date,
                })
        elif ev_type in ("debit", "expense", "payment") or status == "pending":
            if ev_date and ev_date >= request_date:
                pending_debits.append({
                    "event_id": eid,
                    "amount": amount,
                    "date": ev_date,
                    "name": ev.get("name", eid),
                })

    return {
        "balance": balance,
        "min_balance": min_balance,
        "recurring": recurring,
        "confirmed_income": confirmed_income,
        "pending_debits": pending_debits,
        "profile": profile,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. BALANCE FORECASTER
# ─────────────────────────────────────────────────────────────────────────────

def get_recurring_on_date(recurring: list[dict], target_date: date) -> float:
    """Sum of recurring expenses that fall on target_date (simplified monthly)."""
    total = 0.0
    for ev in recurring:
        nd = ev["next_date"]
        if nd is None:
            continue
        freq = ev["frequency"].lower()
        if freq == "monthly" and nd.day == target_date.day and target_date >= nd:
            total += ev["amount"]
        elif freq == "weekly":
            delta = (target_date - nd).days
            if delta >= 0 and delta % 7 == 0:
                total += ev["amount"]
        elif freq == "yearly" and nd.month == target_date.month and nd.day == target_date.day:
            total += ev["amount"]
    return total


def forecast_balance(state: dict, from_date: date, to_date: date,
                     extra_debit_on_date: dict = None) -> dict:
    """
    Simulate balance day-by-day from from_date to to_date.
    Returns: { date: balance }
    extra_debit_on_date: { date: amount } for payment plan debits
    """
    balance = state["balance"]
    balances = {}

    # Index income and pending debits by date
    income_by_date = defaultdict(float)
    for inc in state["confirmed_income"]:
        income_by_date[inc["date"]] += inc["amount"]

    pending_by_date = defaultdict(float)
    for deb in state["pending_debits"]:
        pending_by_date[deb["date"]] += deb["amount"]

    current = from_date
    while current <= to_date:
        # Credits
        balance += income_by_date.get(current, 0.0)
        # Recurring debits
        balance -= get_recurring_on_date(state["recurring"], current)
        # One-time pending debits
        balance -= pending_by_date.get(current, 0.0)
        # Extra debits (payment plan)
        if extra_debit_on_date:
            balance -= extra_debit_on_date.get(current, 0.0)
        balances[current] = balance
        current += timedelta(days=1)

    return balances


def safe_to_pay_on_date(state: dict, request_date: date,
                         requested_amount: float, forecast_days: int = 90) -> float:
    """
    Maximum amount safe to pay on request_date without dropping below min_balance
    through the forecast window.
    """
    end_date = request_date + timedelta(days=forecast_days)
    # Forecast without any payment
    balances = forecast_balance(state, request_date, end_date)
    min_future = min(balances.values()) if balances else state["balance"]
    headroom = min_future - state["min_balance"]
    safe = max(0.0, min(headroom, requested_amount))
    return round(safe, 2)


def earliest_full_payment_date(state: dict, request_date: date,
                                requested_amount: float,
                                desired_date: Optional[date],
                                forecast_days: int = 90) -> Optional[date]:
    """
    First date (on or after request_date) when the user can pay the full amount
    and still stay above min_balance through the rest of the forecast window.
    """
    end_date = request_date + timedelta(days=forecast_days)
    balances_no_payment = forecast_balance(state, request_date, end_date)

    current = request_date
    while current <= end_date:
        # If we paid requested_amount on current, what happens after?
        extra = {current: requested_amount}
        balances_with = forecast_balance(state, request_date, end_date,
                                         extra_debit_on_date=extra)
        if all(v >= state["min_balance"] for v in balances_with.values()):
            return current
        current += timedelta(days=1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. DECISION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def decide(request: dict, state: dict, payment_options: list[dict],
           rate_lookup: dict, data: dict) -> dict:
    """
    Core decision function for one request.
    Returns the output row dict.
    """
    req_id = request["request_id"]
    user_id = request["user_id"]
    home_currency = state["profile"].get("home_currency", "USD")
    req_currency = request.get("currency", home_currency)
    request_date = parse_date(request["request_date"]) or date.today()
    desired_date = parse_date(request.get("desired_completion_date", ""))
    allows_partial = request.get("allows_partial_payment", "false").lower() == "true"

    # Convert requested amount to home currency
    raw_amount = safe_float(request.get("requested_amount", 0))
    requested_amount = convert(raw_amount, req_currency, home_currency,
                                rate_lookup, request_date.isoformat())

    forecast_days = 90
    if desired_date:
        forecast_days = max(90, (desired_date - request_date).days + 30)

    # 1. Safe amount on request_date (no spending changes)
    amount_safe = safe_to_pay_on_date(state, request_date, requested_amount, forecast_days)

    # 2. Earliest full payment date
    earliest_full = earliest_full_payment_date(
        state, request_date, requested_amount, desired_date, forecast_days)

    # 3. Check installment options
    installment_option = None
    req_payment_options = [p for p in payment_options if p["request_id"] == req_id]
    for opt in req_payment_options:
        # Verify each installment payment is feasible
        installments = parse_payment_plan(opt.get("plan", ""))
        if installments and check_plan_feasible(state, request_date, installments, forecast_days):
            installment_option = opt
            break

    # 4. Spending changes (flexible, non-essential only)
    spending_changes = []
    potential_savings = 0.0
    flexible_expenses = [e for e in state["recurring"]
                         if e["is_flexible"] and not e["is_essential"]]
    for fe in sorted(flexible_expenses, key=lambda x: x["amount"], reverse=True):
        if amount_safe >= requested_amount:
            break
        spending_changes.append(f"stop:{fe['event_id']}")
        potential_savings += fe["amount"]
        amount_safe_with_change = safe_to_pay_on_date(
            adjust_state(state, [fe["event_id"]], {}),
            request_date, requested_amount, forecast_days)
        if amount_safe_with_change >= requested_amount:
            amount_safe = requested_amount
            break

    spending_changes = spending_changes[:3]  # max 3

    # ── DECISION TREE ─────────────────────────────────────────────────────────

    if amount_safe >= requested_amount:
        # Can pay in full right now
        status = "affordable_now"
        method = "full_payment"
        plan = "none"
        earliest_str = ""
        changes_str = "none"
        explanation = (f"Balance is sufficient. Safe to pay {home_currency} "
                       f"{requested_amount:.2f} in full today, staying above the "
                       f"{home_currency} {state['min_balance']:.2f} minimum.")

    elif installment_option:
        # Installment plan available and feasible
        status = "affordable_with_plan"
        method = "installments"
        plan = installment_option.get("plan", "none")
        earliest_str = earliest_full.isoformat() if earliest_full else ""
        changes_str = "|".join(spending_changes) if spending_changes else "none"
        explanation = (f"Full payment now would drop below minimum balance. "
                       f"Installment plan available: {plan}.")

    elif allows_partial and amount_safe > 0 and earliest_full and (
            not desired_date or earliest_full <= desired_date):
        # Partial payment: pay safe amount now, rest on earliest_full
        status = "affordable_with_plan"
        method = "partial_payment"
        remaining = round(requested_amount - amount_safe, 2)
        plan = f"{request_date.isoformat()}:{amount_safe:.2f}|{earliest_full.isoformat()}:{remaining:.2f}"
        earliest_str = earliest_full.isoformat()
        changes_str = "|".join(spending_changes) if spending_changes else "none"
        explanation = (f"Partial payment of {home_currency} {amount_safe:.2f} today; "
                       f"remaining {home_currency} {remaining:.2f} on {earliest_full}.")

    elif earliest_full and (not desired_date or earliest_full <= desired_date):
        # Can pay full later
        status = "affordable_later"
        method = "wait"
        plan = "none"
        earliest_str = earliest_full.isoformat()
        changes_str = "|".join(spending_changes) if spending_changes else "none"
        explanation = (f"Cannot safely pay {home_currency} {requested_amount:.2f} today. "
                       f"Earliest safe date: {earliest_full}.")

    else:
        # Not affordable within forecast
        status = "not_affordable"
        method = "not_recommended"
        plan = "none"
        earliest_str = earliest_full.isoformat() if earliest_full else ""
        changes_str = "|".join(spending_changes) if spending_changes else "none"
        explanation = (f"Insufficient funds within the forecast period. "
                       f"Current balance {home_currency} {state['balance']:.2f}, "
                       f"minimum required {home_currency} {state['min_balance']:.2f}.")

    return {
        "request_id": req_id,
        "amount_safe_to_pay": round(min(amount_safe, requested_amount), 2),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan,
        "earliest_date_for_full_payment": earliest_str,
        "spending_changes_needed": changes_str,
        "decision_explanation": explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_payment_plan(plan_str: str) -> list[tuple[date, float]]:
    """Parse 'YYYY-MM-DD:amount|...' into list of (date, amount)."""
    if not plan_str or plan_str.lower() == "none":
        return []
    entries = []
    for entry in plan_str.split("|"):
        parts = entry.strip().split(":")
        if len(parts) == 2:
            d = parse_date(parts[0])
            a = safe_float(parts[1])
            if d:
                entries.append((d, a))
    return entries


def check_plan_feasible(state: dict, request_date: date,
                         installments: list[tuple[date, float]],
                         forecast_days: int) -> bool:
    """Check that a payment plan keeps balance above min_balance at all times."""
    end_date = request_date + timedelta(days=forecast_days)
    extra = defaultdict(float)
    for d, a in installments:
        extra[d] += a
    balances = forecast_balance(state, request_date, end_date,
                                extra_debit_on_date=dict(extra))
    return all(v >= state["min_balance"] for v in balances.values())


def adjust_state(state: dict, stop_event_ids: list[str],
                 reduce_to: dict) -> dict:
    """Return a modified state with flexible expenses stopped or reduced."""
    import copy
    new_state = copy.deepcopy(state)
    new_recurring = []
    for ev in new_state["recurring"]:
        if ev["event_id"] in stop_event_ids:
            continue  # remove
        elif ev["event_id"] in reduce_to:
            ev["amount"] = reduce_to[ev["event_id"]]
        new_recurring.append(ev)
    new_state["recurring"] = new_recurring
    return new_state


# ─────────────────────────────────────────────────────────────────────────────
# 7. AI ENHANCEMENT (optional — uses Claude to improve explanation + edge cases)
# ─────────────────────────────────────────────────────────────────────────────

def enhance_with_ai(row: dict, request: dict, state: dict) -> dict:
    """
    Pass the decision row to Claude for a better explanation and sanity check.
    Only used if USE_AI is True.
    """
    if not USE_AI:
        return row

    prompt = f"""
You are a financial advisor AI. Review this affordability decision and improve the explanation.
Be specific, concise (2-3 sentences), and mention the key financial facts.

Request: {json.dumps(request, default=str)}
User financial state summary:
  balance: {state['balance']}
  min_balance: {state['min_balance']}
  recurring_count: {len(state['recurring'])}
  upcoming_income: {[f"{i['date']}:{i['amount']}" for i in state['confirmed_income'][:3]]}

Current decision:
  status: {row['affordability_status']}
  method: {row['recommended_payment_method']}
  amount_safe: {row['amount_safe_to_pay']}
  plan: {row['payment_plan']}
  explanation: {row['decision_explanation']}

Return ONLY the improved explanation as plain text. No JSON, no preamble.
"""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        row["decision_explanation"] = resp.content[0].text.strip()
    except Exception as e:
        print(f"[WARN] AI enhancement failed for {row['request_id']}: {e}")

    return row


# ─────────────────────────────────────────────────────────────────────────────
# 8. TOKEN USAGE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

usage_log = []  # list of {request_id, input_tokens, output_tokens}


def log_usage(request_id: str, resp):
    if hasattr(resp, "usage"):
        usage_log.append({
            "request_id": request_id,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        })


def write_usage_report():
    os.makedirs("evaluation", exist_ok=True)
    total_in = sum(u["input_tokens"] for u in usage_log)
    total_out = sum(u["output_tokens"] for u in usage_log)
    n = len(usage_log) or 1

    # Claude Sonnet 4.6 pricing (approximate)
    COST_PER_M_IN = 3.0
    COST_PER_M_OUT = 15.0
    total_cost = (total_in / 1_000_000 * COST_PER_M_IN +
                  total_out / 1_000_000 * COST_PER_M_OUT)

    report = f"""# Token Usage Report

## Model
- Provider: Anthropic
- Model: {MODEL}

## Totals
- Total requests processed: {n}
- Total input tokens: {total_in:,}
- Total output tokens: {total_out:,}
- Total tokens: {total_in + total_out:,}
- Estimated total cost: ${total_cost:.4f}

## Per-Request Averages
- Avg input tokens: {total_in // n:,}
- Avg output tokens: {total_out // n:,}
- Avg cost per request: ${total_cost / n:.6f}

## Notes
- Image extraction calls used for blank-amount events
- AI enhancement calls used to improve decision explanations
- Costs are estimates based on public Anthropic pricing
"""
    with open("evaluation/usage_report.md", "w") as f:
        f.write(report)
    print(f"[INFO] Usage report written to evaluation/usage_report.md")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading data...")
    data = load_all_data()

    print("[INFO] Resolving blank amounts from images...")
    data["events"] = resolve_blank_amounts(data["events"], data["images"])

    rate_lookup = build_rate_lookup(data["exchange_rates"])

    requests = data["requests"]
    print(f"[INFO] Processing {len(requests)} requests...")

    results = []

    for i, req in enumerate(requests):
        req_id = req["request_id"]
        user_id = req["user_id"]
        request_date = parse_date(req.get("request_date", "")) or date.today()
        home_currency = next(
            (p["home_currency"] for p in data["profiles"] if p["user_id"] == user_id),
            "USD"
        )

        state = build_user_state(user_id, request_date, data, rate_lookup, home_currency)

        row = decide(req, state, data["payment_options"], rate_lookup, data)

        # Optionally enhance explanation with AI (comment out to save tokens)
        row = enhance_with_ai(row, req, state)

        results.append(row)

        if (i + 1) % 25 == 0:
            print(f"[INFO] Processed {i + 1}/{len(requests)}")

    # Write output
    print(f"[INFO] Writing {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})

    print(f"[INFO] Done! Output written to {OUTPUT_PATH}")

    # Validate
    validate_output(results, requests)

    # Write usage report
    write_usage_report()


def validate_output(results: list[dict], requests: list[dict]):
    print("[INFO] Validating output...")
    req_ids = {r["request_id"] for r in requests}
    out_ids = {r["request_id"] for r in results}
    missing = req_ids - out_ids
    if missing:
        print(f"[ERROR] Missing request_ids in output: {missing}")
    else:
        print(f"[OK] All {len(req_ids)} request_ids present.")

    errors = 0
    for row in results:
        amt = safe_float(row.get("amount_safe_to_pay", 0))
        if amt < 0:
            print(f"[ERROR] Negative amount_safe_to_pay for {row['request_id']}")
            errors += 1
        status_vals = {"affordable_now", "affordable_with_plan",
                       "affordable_later", "not_affordable"}
        if row.get("affordability_status") not in status_vals:
            print(f"[ERROR] Invalid affordability_status for {row['request_id']}")
            errors += 1
    if errors == 0:
        print("[OK] All output rows pass basic validation.")
    else:
        print(f"[WARN] {errors} validation errors found.")


if __name__ == "__main__":
    main()
