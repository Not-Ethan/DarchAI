from services.text_process.ai.taglines.model_loader.bart_base import generate_tagline as bart_base
from services.text_process.ai.taglines.model_loader.t5_small import generate_tagline as t5_small
def load_model(text, model=0):
    if model == 0:
        return bart_base(text)
    elif model == 1:
        return t5_small(text)
    else:
        return t5_small(text)