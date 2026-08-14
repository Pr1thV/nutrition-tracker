"""
Observability and Tracing Setup.
Configures OpenTelemetry instrumentation and exports distributed traces and spans to Langfuse.
"""
import base64
import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)


def init_observability(service_name: str = "NutritionTrackerAI") -> trace.Tracer:
    """
    Initializes the OpenTelemetry TracerProvider with Langfuse OTLP export.
    Falls back gracefully to ConsoleSpanExporter or NoOp if keys are unset.
    """
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        }
    )

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    enable_otel = os.getenv("ENABLE_OTEL_TRACING", "true").lower() == "true"

    if enable_otel and public_key and secret_key:
        try:
            auth_str = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
            otlp_endpoint = f"{langfuse_url.rstrip('/')}/api/public/otel/v1/traces"

            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                headers={
                    "Authorization": f"Basic {auth_str}",
                    "x-langfuse-ingestion-version": "4",
                },
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OpenTelemetry initialized. Exporting traces to Langfuse ({otlp_endpoint})")
        except Exception as e:
            logger.warning(f"Could not initialize Langfuse OTLP exporter: {e}")
    else:
        logger.info("Langfuse credentials not detected. OpenTelemetry initialized with local tracer.")

    return trace.get_tracer(service_name)


# Global tracer instance
tracer = init_observability()
