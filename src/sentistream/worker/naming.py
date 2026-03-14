import logging

from litellm import completion

from sentistream.shared.config import settings


logger = logging.getLogger(settings.app.name)


class ClusterNamer:
    def __init__(self):
        self.model_name = settings.llm.model
        self.api_key = settings.llm.api_key

        # We explicitly instruct the LLM on its behavior to keep responses short and deterministic
        self.system_prompt = (
            "You are an AI assistant analyzing customer review clusters. "
            "Given a list of reviews belonging to the same cluster, generate a "
            "very short (1-4 words), descriptive name summarizing the underlying topic or sentiment. "
            "Examples: 'UI Bugs', 'Great Customer Service', 'Shipping Delays'. "
            "Return ONLY the requested name, nothing else."
        )

    def generate_cluster_name(self, sample_reviews: list[str]) -> str:
        """
        Takes a list of reviews near the centroid of a cluster and asks the
        LLM to infer a short, human-readable name for it.
        """
        if not sample_reviews:
            return "Uncategorized"

        # Combine reviews into a single text block
        reviews_text = "\n".join(f"- {review}" for review in sample_reviews)
        user_message = (
            f"Here are the reviews:\n{reviews_text}\n\nWhat is the short topic name?"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            # LiteLLM automatically routes to the right provider (OpenAI, Anthropic, Ollama)
            # based on the model string prefix (e.g. "gpt-4o", "ollama/llama3")
            response = completion(
                model=self.model_name,
                messages=messages,
                api_key=self.api_key,
                temperature=0.1,  # Low temp for deterministic naming
                max_tokens=15,  # We only want 1-4 words, force cutoff just in case
            )

            # Extract content and strip any unintentional whitespace/quotes
            topic_name = response.choices[0].message.content.strip().strip("'\"")
            return topic_name

        except Exception as e:
            logger.error(f"Failed to generate cluster name via LLM: {e}")
            return "Unknown Topic"
