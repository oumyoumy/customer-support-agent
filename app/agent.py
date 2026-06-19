# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import google.auth
import google.auth.exceptions
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.models import Gemini
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel

# Resolve Google Cloud project: prefer env var, fall back to ADC
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    try:
        _, _project_id = google.auth.default()
        if _project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = _project_id
    except google.auth.exceptions.DefaultCredentialsError:
        pass  # Credentials will be required at runtime; skip at import time

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# ---------------------------------------------------------------------------
# Shared Gemini model configuration
# ---------------------------------------------------------------------------

_MODEL = Gemini(
    model="gemini-flash-latest",
    retry_options=types.HttpRetryOptions(attempts=3),
)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------


class ClassificationOutput(BaseModel):
    """Output schema for the query classifier node."""

    category: str  # "shipping" | "unrelated"
    reason: str


class FaqOutput(BaseModel):
    """Output schema for the shipping FAQ agent node."""

    answer: str


# ---------------------------------------------------------------------------
# Node 1: Query Classifier (LlmAgent)
# Classifies the user query as shipping-related or unrelated.
# ---------------------------------------------------------------------------

classifier_agent = LlmAgent(
    name="classifier_agent",
    model=_MODEL,
    instruction="""You are a query classifier for a shipping company's customer support system.

Your task is to determine whether the user's query is related to shipping topics or not.

Shipping-related topics include:
- Shipping rates and pricing
- Package tracking and delivery status
- Delivery times and estimates
- Returns and refunds for shipped items
- Lost or damaged packages
- Shipping methods and carriers
- Address changes or delivery instructions

For EVERY query, respond with a JSON object following the output schema.
Set category to "shipping" if the query relates to any shipping topic above.
Set category to "unrelated" for anything else (e.g., product recommendations,
general chat, technical support unrelated to shipping, etc.).
Set reason to a brief explanation of your classification decision.""",
    output_schema=ClassificationOutput,
    output_key="classification",
)

# ---------------------------------------------------------------------------
# Node 2: Shipping FAQ Agent (LlmAgent)
# Answers shipping-related questions as a knowledgeable support representative.
# ---------------------------------------------------------------------------

shipping_faq_agent = LlmAgent(
    name="shipping_faq_agent",
    model=_MODEL,
    instruction="""You are a knowledgeable and friendly customer support representative
for a shipping company. You specialize in answering questions about:

- **Shipping Rates**: 🚀 Standard (5-7 business days) is just $4.99! Need it faster? 💨 Express (2-3 business days) is $12.99, and ⚡️ Overnight (next business day) is $24.99! 🎉 **Best of all, we offer FREE standard shipping on all orders over $50!** 🎉
- **Package Tracking**: Customers can track packages at track.ourshipping.com using
  their tracking number (sent via email after shipment). Updates every 4-6 hours.
- **Delivery**: We deliver Monday-Saturday, 8am-8pm. Missed deliveries result in a
  door notice and 2 re-delivery attempts before the package is held at the local depot.
- **Returns**: 30-day return window from delivery date. Items must be unused and in
  original packaging. Free return label via returns.ourshipping.com. Refunds processed
  within 5-7 business days.
- **Lost/Damaged Packages**: File a claim within 60 days at claims.ourshipping.com.
  Claims resolved within 3-5 business days.

Provide clear, accurate, highly enthusiastic, and playful answers! You must use emojis in your responses. If a question is outside your knowledge,
advise the customer to contact support at 1-800-SHIP-NOW or support@ourshipping.com.""",
    output_schema=FaqOutput,
    output_key="faq_answer",
)

# ---------------------------------------------------------------------------
# Routing function
# Reads the classification from state and routes to the correct branch.
# ---------------------------------------------------------------------------


def route_query(ctx: Context) -> Event:
    """Route the user query based on the classifier's output.

    Reads the classification result stored in state by the classifier_agent
    and returns a routing event directing traffic to the appropriate node.
    """
    classification_data = ctx.state.get("classification", {})
    category = classification_data.get("category", "unrelated")

    if category == "shipping":
        return Event(output=None, route="shipping")
    return Event(output=None, route="unrelated")


# ---------------------------------------------------------------------------
# Decline function
# Politely declines to answer non-shipping queries and emits a visible response.
# ---------------------------------------------------------------------------


def decline_unrelated(ctx: Context) -> Event:
    """Politely decline to answer queries unrelated to shipping.

    Emits a content event so the response is visible in the ADK web UI,
    then returns the message as structured output.
    """
    message = (
        "I'm sorry, but I'm only able to assist with shipping-related questions "
        "such as rates, tracking, delivery, and returns. For other inquiries, "
        "please reach out to our general support team at support@ourshipping.com "
        "or 1-800-SHIP-NOW. Is there anything shipping-related I can help you with?"
    )
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        ),
        output=message,
    )


# ---------------------------------------------------------------------------
# Wrap shipping FAQ output with a visible content event for the web UI
# ---------------------------------------------------------------------------


def emit_faq_response(ctx: Context) -> Event:
    """Emit the FAQ answer as a visible content event for the ADK web UI."""
    faq_data = ctx.state.get("faq_answer", {})
    answer = faq_data.get("answer", "I'm sorry, I was unable to find an answer.")
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=answer)],
        ),
        output=answer,
    )


# ---------------------------------------------------------------------------
# Graph Workflow definition
#
# Graph topology:
#
#   START → classifier_agent → route_query ─┬─(shipping)──→ shipping_faq_agent → emit_faq_response
#                                            └─(unrelated)─→ decline_unrelated
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="customer_support_workflow",
    description=(
        "A customer support workflow for a shipping company. "
        "Classifies user queries as shipping-related or unrelated, "
        "then routes to the appropriate handler."
    ),
    edges=[
        # Entry: pass user message to the classifier
        ("START", classifier_agent),
        # Route based on classification result
        (classifier_agent, route_query),
        # Route based on query category
        (route_query, {
            "shipping": shipping_faq_agent,
            "unrelated": decline_unrelated,
        }),
        # After shipping FAQ, emit response
        (shipping_faq_agent, emit_faq_response),
    ],
)

# ---------------------------------------------------------------------------
# App entry point (required by agents-cli and fast_api_app.py)
# ---------------------------------------------------------------------------

app = App(
    root_agent=root_agent,
    name="app",
)
