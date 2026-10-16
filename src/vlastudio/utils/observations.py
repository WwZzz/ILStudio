"""Representation hints carried in the standard sample's reasoning metadata."""


def is_feature_image(sample):
    reasoning = sample.get('reasoning')
    if isinstance(reasoning, list):
        return bool(reasoning) and all(is_feature_image({'reasoning': row}) for row in reasoning)
    return isinstance(reasoning, dict) and reasoning.get('image_type') == 'features'
