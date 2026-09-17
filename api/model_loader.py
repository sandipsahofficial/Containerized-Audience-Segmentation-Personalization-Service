import os
import joblib
import logging

logger = logging.getLogger('api.model_loader')

class ModelLoader:
    _instance = None
    _bundle = None

    @classmethod
    def get_bundle_path(cls):
        env_dir = os.environ.get('MODELS_DIR')
        if env_dir:
            p = os.path.join(env_dir, 'model_bundle.joblib')
            return p if os.path.exists(p) else None

        candidate_paths = [
            'models/model_bundle.joblib',
            '/app/models/model_bundle.joblib',
            '../models/model_bundle.joblib'
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return p
        return None

    @classmethod
    def load_model(cls, force_reload=False):
        if cls._bundle is not None and not force_reload:
            return cls._bundle

        path = cls.get_bundle_path()
        if not path:
            logger.warning('Model bundle file does not exist on disk.')
            return None

        try:
            logger.info(f'Loading model artifact bundle from: {path}')
            cls._bundle = joblib.load(path)
            algo = cls._bundle.get('algorithm')
            k = cls._bundle.get('n_clusters')
            logger.info(f'Model loaded successfully: algorithm={algo}, n_clusters={k}')
            return cls._bundle
        except Exception as e:
            logger.error(f'Failed to load model artifact: {e}')
            return None

    @classmethod
    def is_loaded(cls):
        if cls._bundle is not None:
            return True
        return cls.load_model() is not None

    @classmethod
    def get_bundle(cls):
        if cls._bundle is None:
            return cls.load_model()
        return cls._bundle
