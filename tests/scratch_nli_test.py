from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
print("Labels:", getattr(model.model.config, "id2label", None))

scores = model.predict([
    ("We have logged all potential system hazards and mitigations.", "This document contains a risk register."),
    ("The AI system is a chatbot for customer service.", "This document contains a risk register.")
])

print("Scores:", scores)
