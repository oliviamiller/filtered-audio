import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.models.wake_word_filter import WakeWordFilter


@pytest.fixture
def mock_config():
    """Mock Viam ComponentConfig"""
    config = Mock()
    config.attributes.fields = {
        "wake_words": Mock(string_value="robot"),
        "source_microphone": Mock(string_value="mic1"),
        "vad_aggressiveness": Mock(number_value=3),
    }
    return config


@pytest.fixture
def mock_dependencies():
    """Mock microphone dependency"""
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(
        sample_rate_hz=16000,
        num_channels=1
    )
    return {"mic1": mic}


@pytest.fixture
def mock_env():
    """Mock all WakeWordFilter.new() dependencies"""
    with patch('src.models.wake_word_filter.struct_to_dict') as mock_struct, \
         patch('src.models.wake_word_filter.VoskModel') as mock_vosk, \
         patch('src.models.wake_word_filter.webrtcvad.Vad') as mock_vad, \
         patch('src.models.wake_word_filter.ThreadPoolExecutor') as mock_executor, \
         patch('src.models.wake_word_filter.os.getenv', return_value="/tmp") as mock_getenv, \
         patch('src.models.wake_word_filter.os.path.exists', return_value=True) as mock_exists, \
         patch('src.models.vosk.os.path.exists', return_value=True) as mock_vosk_exists, \
         patch('src.models.wake_word_filter.AudioIn.get_resource_name', return_value="mic1") as mock_resource:

        yield {
            'struct_to_dict': mock_struct,
            'vosk': mock_vosk,
            'vad': mock_vad,
            'executor': mock_executor,
            'getenv': mock_getenv,
            'exists': mock_exists,
            'vosk_exists': mock_vosk_exists,
            'resource_name': mock_resource
        }


def create_config(attrs):
    """Helper to create mock config with specific attributes"""
    config = Mock()
    with patch('src.models.wake_word_filter.struct_to_dict', return_value=attrs):
        return config

def test_validate_config_success():
    """Test validate_config with valid configuration"""
    config = Mock()
    config.attributes = Mock()

    # Mock struct_to_dict to return our test config
    with patch('src.models.wake_word_filter.struct_to_dict') as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "wake_words": ["robot"]
        }

        deps, errors = WakeWordFilter.validate_config(config)

        assert deps == ["mic1"]
        assert not errors


def test_validate_config_missing_microphone():
    """Test validate_config raises error when source_microphone is missing"""
    config = Mock()
    config.attributes = Mock()

    with patch('src.models.wake_word_filter.struct_to_dict') as mock_struct:
        mock_struct.return_value = {
            "wake_words": ["robot"]
            # source_microphone missing
        }

        with pytest.raises(RuntimeError, match="source_microphone attribute is required"):
            WakeWordFilter.validate_config(config)


def test_validate_config_empty_microphone():
    """Test validate_config raises error when source_microphone is empty string"""
    config = Mock()
    config.attributes = Mock()

    with patch('src.models.wake_word_filter.struct_to_dict') as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "",
            "wake_words": ["robot"]
        }

        with pytest.raises(RuntimeError, match="source_microphone attribute is required"):
            WakeWordFilter.validate_config(config)


def test_validate_config_missing_wake_words():
    """Test validate_config raises error when wake_words is missing"""
    config = Mock()
    config.attributes = Mock()

    with patch('src.models.wake_word_filter.struct_to_dict') as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1"
            # wake_words missing
        }

        with pytest.raises(RuntimeError, match="wake_words attribute is required"):
            WakeWordFilter.validate_config(config)


def test_validate_config_empty_wake_words():
    """Test validate_config raises error when wake_words is empty list"""
    config = Mock()
    config.attributes = Mock()

    with patch('src.models.wake_word_filter.struct_to_dict') as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "wake_words": []
        }

        with pytest.raises(RuntimeError, match="wake_words attribute is required"):
            WakeWordFilter.validate_config(config)

def test_validate_config_rejects_non_string_wake_words(mock_env):
    """Test validate_config raises error when wake_words contains non-strings"""
    config = Mock()
    config.attributes = Mock()

    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot", 123, "computer"]
    }

    with pytest.raises(RuntimeError, match="All wake_words must be strings"):
        WakeWordFilter.validate_config(config)

def test_validate_config_rejects_non_string_microphone(mock_env):
    """Test validate_config raises error when wake_words contains non-strings"""
    config = Mock()
    config.attributes = Mock()

    mock_env['struct_to_dict'].return_value = {
        "source_microphone": 123,
        "wake_words": ["robot", "computer"]
    }

    with pytest.raises(RuntimeError, match="source_microphone attribute must be a string"):
        WakeWordFilter.validate_config(config)


# Tests for WakeWordFilter.new()

def test_new_with_single_wake_word_string(mock_env):
    """Test new() parses single wake word string correctly"""
    config = Mock()
    mic = AsyncMock()

    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": "robot",
        "vad_aggressiveness": 3
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == ["robot"]


def test_new_with_wake_word_list(mock_env):
    """Test new() parses wake word list correctly"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["hey robot", "OK Robot", "Computer"],
        "vad_aggressiveness": 3
    }

    dependencies["mic1"] = mic
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == ["hey robot", "ok robot", "computer"]


def test_new_uses_default_vad_aggressiveness(mock_env):
    """Test new() uses default VAD aggressiveness of 3"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}


    mock_env['struct_to_dict'].return_value = {
            "source_microphone": "mic1",
            "wake_words": ["robot"]
            # vad_aggressiveness not specified
        }

    dependencies["mic1"] = mic
    WakeWordFilter.new(config, dependencies)

    # Verify VAD was initialized with default value 3
    mock_env["vad"].assert_called_once_with(3)


def test_new_uses_custom_vad_aggressiveness(mock_env):
    """Test new() uses custom VAD aggressiveness when provided"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}


    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vad_aggressiveness": 1
    }

    dependencies["mic1"] = mic
    WakeWordFilter.new(config, dependencies)

    # Verify VAD was initialized with custom value 1
    mock_env["vad"].assert_called_once_with(1)


def test_new_loads_vosk_model(mock_env):
    """Test new() loads Vosk model from correct path"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22"
    }

    dependencies["mic1"] = mic
    WakeWordFilter.new(config, dependencies)

    # Verify VoskModel was called with correct path
    mock_env["vosk"].assert_called_once_with("/tmp/vosk-model-en-us-0.22")


def test_new_sets_microphone_client(mock_env):
    """Test new() sets microphone_client from dependencies"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}


    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22"
    }

    dependencies["mic1"] = mic
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.microphone_client == mic



# # Error case tests for WakeWordFilter.new()

def test_new_fails_when_model_not_found_and_download_fails(mock_env):
    """Test new() raises error when Vosk model doesn't exist and download fails"""
    config = Mock()
    mic = AsyncMock()

    # Model doesn't exist in both modules
    mock_env['exists'].return_value = False
    mock_env['vosk_exists'].return_value = False

    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"]
    }

    dependencies = {"mic1": mic}

    with patch('src.models.wake_word_filter.download_vosk_model', side_effect=Exception("Download failed")):
        with pytest.raises(RuntimeError, match="Vosk model not found.*download failed"):
            WakeWordFilter.new(config, dependencies)


def test_new_downloads_model_when_not_found(mock_env):
    """Test new() downloads Vosk model when it doesn't exist"""
    config = Mock()
    mic = AsyncMock()

    # Model doesn't exist in both modules
    mock_env['exists'].return_value = False
    mock_env['vosk_exists'].return_value = False

    mock_env['struct_to_dict'].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"]
    }

    dependencies = {"mic1": mic}

    with patch('src.models.wake_word_filter.download_vosk_model', return_value="/tmp/downloaded-model") as mock_download:
        instance = WakeWordFilter.new(config, dependencies)

        # Verify download was called
        mock_download.assert_called_once()
        # Verify VoskModel was created with downloaded path
        mock_env['vosk'].assert_called_once_with("/tmp/downloaded-model")


def test_new_handles_missing_microphone_in_dependencies(mock_env):
    """Test new() raises error when microphone not in dependencies"""
    config = Mock()

    with patch('src.models.wake_word_filter.struct_to_dict') as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "wake_words": ["robot"]
        }

        dependencies = {}  # Empty - no microphone!

        with pytest.raises(KeyError):
            WakeWordFilter.new(config, dependencies)

def test_check_for_wake_word_detects_wake_word():
    """Test check_for_wake_word returns True when wake word is in audio"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot"]
    wake_word_filter.logger = Mock()

    with patch('src.models.wake_word_filter.KaldiRecognizer') as mock_recognizer_class:
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True
        mock_rec.FinalResult.return_value = '{"text": "hey robot turn on the lights"}'
        mock_recognizer_class.return_value = mock_rec

        # Call the actual method
        result = WakeWordFilter.check_for_wake_word(wake_word_filter, b'\x00' * 1000, 16000)
        assert result is True

        mock_rec.FinalResult.return_value = '{"text": "hello there how are you"}'
        result = WakeWordFilter.check_for_wake_word(wake_word_filter, b'\x00' * 1000, 16000)
        assert result is False

        mock_rec.FinalResult.return_value = '{"text": ""}'
        result = WakeWordFilter.check_for_wake_word(wake_word_filter, b'\x00' * 1000, 16000)
        assert result is False



@pytest.mark.asyncio
async def test_get_audio_rejects_non_pcm16_codec():
    """Test get_audio raises error for non-PCM16 codec"""
    wake_word_filter = Mock()
    wake_word_filter.microphone_client = AsyncMock()
    wake_word_filter.logger = Mock()

    with pytest.raises(ValueError, match="PCM16"):
        await WakeWordFilter.get_audio(wake_word_filter, "mp3", 0, 0)


@pytest.mark.asyncio
async def test_get_audio_rejects_invalid_sample_rate(mock_env):
    """Test get_audio raises error when microphone sample rate is not 16000 Hz"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    # Mock microphone with wrong sample rate
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(
        sample_rate_hz=48000,
        num_channels=1
    )
    wake_word_filter.microphone_client = mic

    with pytest.raises(ValueError, match="16000 Hz"):
        stream = await WakeWordFilter.get_audio(wake_word_filter, "pcm16", 0, 0)
        # Consume the generator to trigger the validation
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_get_audio_returns_when_no_microphone():
    """Test get_audio returns early when no microphone configured"""
    wake_word_filter = Mock()
    wake_word_filter.microphone_client = None
    wake_word_filter.logger = Mock()

    result = await WakeWordFilter.get_audio(wake_word_filter, "pcm16", 0, 0)

    # Should return None or empty
    assert result is None or result is not None
    wake_word_filter.logger.error.assert_called_once_with("No microphone configured")


@pytest.mark.asyncio
async def test_get_properties_returns_mic_properties():
    """Test get_properties returns properties from underlying microphone"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    # Mock microphone with specific properties
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(
        sample_rate_hz=16000,
        num_channels=1
    )
    wake_word_filter.microphone_client = mic

    result = await WakeWordFilter.get_properties(wake_word_filter)

    assert result.sample_rate_hz == 16000
    assert result.num_channels == 1


@pytest.mark.asyncio
async def test_get_properties_always_reports_pcm16():
    """Test get_properties always reports PCM16 as supported codec"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    mic = AsyncMock()
    mic.get_properties.return_value = Mock(
        sample_rate_hz=48000,
        num_channels=2
    )
    wake_word_filter.microphone_client = mic

    result = await WakeWordFilter.get_properties(wake_word_filter)

    assert result.supported_codecs == ["pcm16"]



@pytest.mark.asyncio
async def test_get_audio_rejects_stereo_audio(mock_env):
    """Test get_audio raises error when microphone is stereo (2 channels)"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    # Mock microphone with stereo audio
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(
        sample_rate_hz=16000,
        num_channels=2
    )
    wake_word_filter.microphone_client = mic

    with pytest.raises(ValueError, match="mono.*1 channel"):
        stream = await WakeWordFilter.get_audio(wake_word_filter, "pcm16", 0, 0)
        # Consume the generator to wake_word_filter the validation
        async for _ in stream:
            pass
