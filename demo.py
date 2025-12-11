import torch

# --- Assume you already have these three models ready ---

# harmful_model: takes text → returns prob_harmful in [0,1]
# topic_model:   takes text → returns topic_id or topic label
# suggestor:     takes (text, topic, harmful_prob) → returns suggestion string

harmful_threshold = 0.5   # tune this

def classify_harmful(text):
    """Return float probability and binary label."""
    with torch.no_grad():
        # example: logits = harmful_model(text)
        logits = harmful_model(text)          # shape [1,2] or [2]
        probs = torch.softmax(logits, dim=-1)
        p_harm = float(probs[1])             # index 1 = harmful
    return p_harm, int(p_harm >= harmful_threshold)

def classify_topic(text):
    """Return topic id/label."""
    with torch.no_grad():
        logits = topic_model(text)           # shape [1, num_topics]
        topic_id = int(torch.argmax(logits, dim=-1))
    return topic_id

def suggest_action(text, topic_id, p_harm):
    """Call your suggestor (could be an LM or rule-based)."""
    return suggestor(text=text, topic_id=topic_id, harmful_prob=p_harm)

# --------- MAIN PIPELINE FUNCTION ---------

def process_text(text):
    """
    1) Run harmful classifier
    2) If not harmful → run topic classifier
    3) Always call suggestor with results
    """
    p_harm, is_harm = classify_harmful(text)

    if is_harm:
        topic_id = None      # or a special "HARMFUL" topic
    else:
        topic_id = classify_topic(text)

    suggestion = suggest_action(text, topic_id, p_harm)

    return {
        "text": text,
        "harmful_prob": p_harm,
        "harmful_label": is_harm,
        "topic_id": topic_id,
        "suggestion": suggestion,
    }

# --------- EXAMPLE USAGE ---------

example = "Some input text here..."
result = process_text(example)
print(result)
