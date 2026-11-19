# AI Customer Support Agent

An email support agent that reads a Gmail inbox, classifies each message, drafts a reply from an approved knowledge base, and sends the reply or hands the thread to a person.

The models propose. Python decides what is allowed to go out.

## What it does

A poller checks Gmail on a timer for mail that is still in the inbox and not labeled `ai-processed`. For each message it:

1. Skips mail the agent must not answer: its own sent mail, auto-replies, and bulk or list mail.
2. Saves the message on a ticket in PostgreSQL.
3. Classifies the thread into a section and a structured decision.
4. Drafts a reply with a tool that reads one section of the knowledge base.
5. Sends the reply, or labels the thread `needs-human` and stores a proposed next step for a person.

Replies are sent only for steps the customer can do themselves, such as checking a valve or sending a missing detail. Refunds, shipments, exchanges, attachments, prompt-injection attempts, and replies that promise an action are escalated.

## Stack

- Python 3.13, uv, FastAPI, Pydantic AI
- Anthropic Claude: Haiku classifies, Sonnet drafts
- Gmail API for reading, labeling, and sending
- PostgreSQL on Supabase for events, tickets, messages, and the knowledge base
- React, TypeScript, Vite, and Tailwind for the admin dashboard
- Docker Compose for the API and the poller
- pydantic-evals for a golden-email evaluation set

## How a decision is made

`app/ai/decide.py` runs one thread:

- The classifier returns a section, a reason, and whether a person is required.
- Phrase rules run first. The model is called only when those rules do not match.
- The reply agent may call `get_section` and must return structured output. It cannot send mail or take an action.
- Python checks the draft. If the next step is not a customer step, the draft lacks information, or the draft promises an action, the thread is escalated.

## Project layout

```
app/poller.py            Inbox loop
app/pipeline.py          One message: save, decide, reply or escalate
app/ai/classifier.py     Section, action, and phrase rules
app/ai/agent.py          Reply draft and knowledge-base tool
app/ai/decide.py         Final send-or-escalate decision
app/ai/rules.py          Product rules for the next step
app/gmail.py             Gmail read, send, and labels
app/main.py              Health check, JSON API, and dashboard files
dashboard/               Admin UI for events, tickets, and the knowledge base
knowledge_base/          Source markdown seeded into PostgreSQL
evals/                   Golden emails and section/action checks
supabase/migrations/     Table definitions
```

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the project root:

```bash
uv sync
copy .env.example .env
```

On macOS or Linux, use `cp .env.example .env`.

Fill in `.env`. Never commit it.

| Variable | Purpose |
| --- | --- |
| `GOOGLE_CLIENT_ID` | Gmail OAuth client |
| `GOOGLE_CLIENT_SECRET` | Gmail OAuth secret |
| `GOOGLE_REFRESH_TOKEN` | Gmail access for the mailbox |
| `DATABASE_URL` | PostgreSQL connection string |
| `ANTHROPIC_API_KEY` | Claude API key |
| `CLASSIFIER_MODEL` | Defaults to `anthropic:claude-haiku-5-5` |
| `AGENT_MODEL` | Defaults to `anthropic:claude-sonnet-5-5` |
| `POLL_INTERVAL_SECONDS` | Seconds between inbox checks |
| `DASHBOARD_USERNAME` | Admin login, default `admin` |
| `DASHBOARD_PASSWORD` | Admin password |

Create a Google OAuth desktop client, download it as `credentials.json` in the project root, and add the mailbox as a test user on the consent screen. Then:

```bash
uv run python scripts/get_refresh_token.py
```

Put the printed refresh token in `GOOGLE_REFRESH_TOKEN`.

Apply `supabase/migrations/20261103130506_create_tables.sql` in the Supabase SQL editor, then load the knowledge base:

```bash
uv run python scripts/seed_knowledge_base.py
```

## Run

Start the poller. This is the part that reads mail and can call Claude.

```bash
uv run python -m app.poller
```

Start the API:

```bash
uv run fastapi dev app/main.py --host 127.0.0.1 --port 8000
```

`GET /health` is public. Everything under `/api` requires HTTP Basic login.

Dashboard, in a second terminal:

```bash
cd dashboard
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`. The UI has three tabs: Events, Tickets, and Knowledge base. Editing a knowledge-base entry updates the text the agent reads on the next email.

Run one poller only. A second poller will process the same mailbox at the same time.

## Docker

The image builds the dashboard, then runs the Python app. Compose starts two containers from that image: `api` on `127.0.0.1:8080`, and `poller` with no published port.

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f poller
```

Stop both when you are not using them. A stopped poller does not call the model.

```bash
docker compose stop
docker compose start
```

`docker compose restart` does not reload `.env`. After changing `.env`, recreate the containers:

```bash
docker compose up -d --force-recreate
```

## Evaluation

The golden set checks the section and the action for fixed emails. It does not score reply wording.

```bash
uv run python -m evals.run_evals
```

## Labels

| Label | Meaning |
| --- | --- |
| `ai-processed` | The poller has finished this message, or it failed three times |
| `needs-human` | A person owns the thread until the label is removed |
