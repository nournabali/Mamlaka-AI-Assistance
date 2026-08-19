from __future__ import annotations

from mamlaka_ai.config import PROJECT_ROOT, settings


def test_required_ui_images_are_local() -> None:
    required = {
        "almamlaka-tv-logo.png",
        "mamlaka-ai-avatar.png",
        "mamlaka-ai-sidebar-logo-v3.png",
        "mamlaka-user-avatar.png",
    }
    assert required <= {path.name for path in settings.assets_dir.glob("*.png")}


def test_runtime_has_no_remote_logo_configuration() -> None:
    config_source = (PROJECT_ROOT / "src" / "mamlaka_ai" / "config.py").read_text(
        encoding="utf-8"
    )
    app_source = (
        PROJECT_ROOT / "src" / "mamlaka_ai" / "ui" / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    assert "LOGO_URL" not in config_source
    assert "settings.logo_url" not in app_source
    assert "seeklogo.com" not in config_source + app_source
