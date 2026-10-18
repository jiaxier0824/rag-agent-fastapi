import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.service import AgentService
from config.settings import Settings


class SettingsTest(unittest.TestCase):
    def test_dotenv_key_reaches_both_models_without_export(self):
        # 使用临时假密钥，只构造模型，不发送网络请求。
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("DASHSCOPE_API_KEY=test-dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=env_file)
                with patch("agent.service.settings", settings):
                    service = AgentService(memory_store=None)
                for model in (service.model, service.fallback_model):
                    self.assertEqual(
                        model.dashscope_api_key.get_secret_value(), "test-dotenv-key",
                    )
                self.assertNotIn("test-dotenv-key", repr(settings))
                self.assertNotIn("DASHSCOPE_API_KEY", os.environ)

    def test_environment_key_overrides_dotenv(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("DASHSCOPE_API_KEY=test-dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-env-key"}):
                settings = Settings(_env_file=env_file)
            self.assertEqual(settings.dashscope_api_key.get_secret_value(), "test-env-key")
