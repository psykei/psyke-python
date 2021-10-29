class FeatureNotFoundException(Exception):

    def __init__(self, feature: str):
        super().__init__('Feature "' + feature + '" not found.')
