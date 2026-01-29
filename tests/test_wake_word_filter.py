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
    mic.get_properties.return_value = Mock(sample_rate_hz=16000, num_channels=1)
    return {"mic1": mic}


@pytest.fixture
def mock_env():
    """Mock all WakeWordFilter.new() dependencies"""
    with (
        patch("src.models.wake_word_filter.struct_to_dict") as mock_struct,
        patch("src.models.wake_word_filter.VoskModel") as mock_vosk,
        patch("src.models.wake_word_filter.webrtcvad.Vad") as mock_vad,
        patch("src.models.wake_word_filter.ThreadPoolExecutor") as mock_executor,
        patch(
            "src.models.wake_word_filter.get_vosk_model", return_value="/tmp/vosk-model"
        ) as mock_get_vosk,
        patch(
            "src.models.wake_word_filter.AudioIn.get_resource_name", return_value="mic1"
        ) as mock_resource,
    ):
        yield {
            "struct_to_dict": mock_struct,
            "vosk": mock_vosk,
            "vad": mock_vad,
            "executor": mock_executor,
            "get_vosk_model": mock_get_vosk,
            "resource_name": mock_resource,
        }


def create_config(attrs):
    """Helper to create mock config with specific attributes"""
    config = Mock()
    with patch("src.models.wake_word_filter.struct_to_dict", return_value=attrs):
        return config


def test_validate_config_success():
    """Test validate_config with valid configuration"""
    config = Mock()
    config.attributes = Mock()

    # Mock struct_to_dict to return our test config
    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "wake_words": ["robot"],
        }

        deps, errors = WakeWordFilter.validate_config(config)

        assert deps == ["mic1"]
        assert not errors


def test_validate_config_missing_microphone():
    """Test validate_config raises error when source_microphone is missing"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "wake_words": ["robot"]
            # source_microphone missing
        }

        with pytest.raises(ValueError, match="source_microphone attribute is required"):
            WakeWordFilter.validate_config(config)


def test_validate_config_empty_microphone():
    """Test validate_config raises error when source_microphone is empty string"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {"source_microphone": "", "wake_words": ["robot"]}

        with pytest.raises(ValueError, match="source_microphone attribute is required"):
            WakeWordFilter.validate_config(config)


def test_validate_config_accepts_missing_wake_words():
    """Test validate_config accepts missing wake_words (all speech mode)"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1"
            # wake_words missing - should be valid
        }

        deps, errors = WakeWordFilter.validate_config(config)
        assert deps == ["mic1"]
        assert not errors


def test_validate_config_accepts_empty_wake_words():
    """Test validate_config accepts empty wake_words list (all speech mode)"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {"source_microphone": "mic1", "wake_words": []}

        deps, errors = WakeWordFilter.validate_config(config)
        assert deps == ["mic1"]
        assert not errors


def test_validate_config_rejects_non_string_wake_words(mock_env):
    """Test validate_config raises error when wake_words contains non-strings"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot", 123, "computer"],
    }

    with pytest.raises(ValueError, match="All wake_words must be strings"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_string_microphone(mock_env):
    """Test validate_config raises error when microphone is non-string"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": 123,
        "wake_words": ["robot", "computer"],
    }

    with pytest.raises(
        ValueError, match="source_microphone attribute must be a string"
    ):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_invalid_vad_aggressiveness(mock_env):
    """Test validate_config raises error when vad_agressiveness invalid"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "vad_aggressiveness": 4,
    }

    with pytest.raises(ValueError, match="vad_aggressiveness must be 0-3, got 4"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_vad_agressiveness(mock_env):
    """Test validate_config raises error when vad_agressiveness not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "vad_aggressiveness": "4",
    }

    with pytest.raises(ValueError, match="vad_aggressiveness must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_fuzzy_threshold(mock_env):
    """Test validate_config raises error when fuzzy_threshold not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "fuzzy_threshold": "5",
    }

    with pytest.raises(ValueError, match="fuzzy_threshold must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_invalid_fuzzy_threshold(mock_env):
    """Test validate_config raises error when fuzzy_threshold not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "fuzzy_threshold": 7,
    }

    with pytest.raises(ValueError, match="fuzzy_threshold must be 0-5, got 7"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_silence_duration_ms(mock_env):
    """Test validate_config raises error when silence_duration_ms not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "silence_duration_ms": "500",
    }

    with pytest.raises(ValueError, match="silence_duration_ms must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_positive_silence_duration_ms(mock_env):
    """Test validate_config raises error when silence_duration_ms is not positive"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "silence_duration_ms": 0,
    }

    with pytest.raises(ValueError, match="silence_duration_ms must be positive"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_min_speech_ms(mock_env):
    """Test validate_config raises error when min_speech_ms not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "min_speech_ms": "300",
    }

    with pytest.raises(ValueError, match="min_speech_ms must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_positive_min_speech_ms(mock_env):
    """Test validate_config raises error when min_speech_ms is not positive"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "min_speech_ms": -100,
    }

    with pytest.raises(ValueError, match="min_speech_ms must be positive"):
        WakeWordFilter.validate_config(config)

def test_validate_config_accepts_valid_silence_duration_ms(mock_env):
    """Test validate_config accepts valid silence_duration_ms"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "silence_duration_ms": 1000,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_accepts_valid_min_speech_ms(mock_env):
    """Test validate_config accepts valid min_speech_ms"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "min_speech_ms": 500,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_non_bool_use_grammar(mock_env):
    """Test validate_config raises error when use_grammar is not a boolean"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "use_grammar": "true",
    }

    with pytest.raises(ValueError, match="use_grammar must be a boolean"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_use_grammar_true(mock_env):
    """Test validate_config accepts use_grammar=True"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "use_grammar": True,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_accepts_use_grammar_false(mock_env):
    """Test validate_config accepts use_grammar=False"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "use_grammar": False,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_non_number_grammar_confidence(mock_env):
    """Test validate_config raises error when vosk_grammar_confidence not a number"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": "0.7",
    }

    with pytest.raises(ValueError, match="vosk_grammar_confidence must be a number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_grammar_confidence_too_low(mock_env):
    """Test validate_config raises error when vosk_grammar_confidence below 0"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": -0.1,
    }

    with pytest.raises(ValueError, match="vosk_grammar_confidence must be 0.0-1.0"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_grammar_confidence_too_high(mock_env):
    """Test validate_config raises error when vosk_grammar_confidence above 1"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": 1.5,
    }

    with pytest.raises(ValueError, match="vosk_grammar_confidence must be 0.0-1.0"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_valid_grammar_confidence(mock_env):
    """Test validate_config accepts valid vosk_grammar_confidence"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": 0.8,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


# Tests for WakeWordFilter.new()


def test_new_with_single_wake_word_string(mock_env):
    """Test new() parses single wake word string correctly"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": "robot",
        "vad_aggressiveness": 3,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == ["robot"]


def test_new_with_wake_word_list(mock_env):
    """Test new() parses wake word list correctly"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["hey robot", "OK Robot", "Computer"],
        "vad_aggressiveness": 3,
    }

    dependencies["mic1"] = mic
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == ["hey robot", "ok robot", "computer"]


def test_new_with_empty_wake_words(mock_env):
    """Test new() with empty wake_words enables all-speech mode"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": [],
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == []


def test_new_with_missing_wake_words(mock_env):
    """Test new() with missing wake_words enables all-speech mode"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        # wake_words not specified
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == []


def test_new_uses_default_vad_aggressiveness(mock_env):
    """Test new() uses default VAD aggressiveness of 3"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
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

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vad_aggressiveness": 1,
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

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22",
    }

    dependencies["mic1"] = mic
    WakeWordFilter.new(config, dependencies)

    # Verify get_vosk_model was called with the model name
    mock_env["get_vosk_model"].assert_called_once()
    # Verify VoskModel was called with the path returned by get_vosk_model
    mock_env["vosk"].assert_called_once_with("/tmp/vosk-model")


def test_new_sets_microphone_client(mock_env):
    """Test new() sets microphone_client from dependencies"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22",
    }

    dependencies["mic1"] = mic
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.microphone_client == mic


def test_new_uses_default_silence_duration_ms(mock_env):
    """Test new() uses default silence_duration_ms when not specified"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    # Default is 900ms
    assert instance.silence_duration_ms == 900


def test_new_uses_custom_silence_duration_ms(mock_env):
    """Test new() uses custom silence_duration_ms when provided"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "silence_duration_ms": 1200,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.silence_duration_ms == 1200


def test_new_uses_default_min_speech_duration_ms(mock_env):
    """Test new() uses default min_speech_duration_ms when not specified"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    # Default is 300ms
    assert instance.min_speech_duration_ms == 300


def test_new_uses_custom_min_speech_duration_ms(mock_env):
    """Test new() uses custom min_speech_ms when provided"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "min_speech_ms": 500,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.min_speech_duration_ms == 500


def test_new_uses_default_use_grammar(mock_env):
    """Test new() uses default use_grammar=True when not specified"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    # Default is True (grammar mode)
    assert instance.use_grammar is True


def test_new_uses_custom_use_grammar_false(mock_env):
    """Test new() uses custom use_grammar=False when provided"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "use_grammar": False,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.use_grammar is False


def test_new_uses_default_grammar_confidence(mock_env):
    """Test new() uses defaultvosk_grammar_confidence when not specified"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    # Default is 0.7
    assert instance.grammar_confidence == 0.7


def test_new_uses_custom_grammar_confidence(mock_env):
    """Test new() uses custom vosk_grammar_confidence when provided"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": 0.85,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.grammar_confidence == 0.85


# # Error case tests for WakeWordFilter.new()


def test_new_fails_when_get_vosk_model_fails(mock_env):
    """Test new() raises error when get_vosk_model fails"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    # Make get_vosk_model raise an error
    mock_env["get_vosk_model"].side_effect = RuntimeError("Model not found")

    dependencies = {"mic1": mic}

    with pytest.raises(RuntimeError, match="Model not found"):
        WakeWordFilter.new(config, dependencies)


def test_new_calls_get_vosk_model_with_model_name(mock_env):
    """Test new() calls get_vosk_model with the configured model name"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22",
    }

    dependencies = {"mic1": mic}
    WakeWordFilter.new(config, dependencies)

    # Verify get_vosk_model was called with the model name
    call_args = mock_env["get_vosk_model"].call_args
    assert call_args[0][0] == "vosk-model-en-us-0.22"


def test_new_handles_missing_microphone_in_dependencies(mock_env):
    """Test new() raises error when microphone not in dependencies"""
    config = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {}  # Empty - no microphone!

    with pytest.raises(KeyError):
        WakeWordFilter.new(config, dependencies)


def test_check_for_wake_word_detects_wake_word_anywhere():
    """Test check_for_wake_word returns True when wake word is anywhere in audio"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot"]
    wake_word_filter.fuzzy_matcher = None
    wake_word_filter.use_grammar = True
    wake_word_filter.logger = Mock()

    with patch("src.models.wake_word_filter.KaldiRecognizer") as mock_recognizer_class:
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True

        # Wake word at start - should match
        mock_rec.FinalResult.return_value = '{"text": "robot turn on the lights"}'
        mock_recognizer_class.return_value = mock_rec
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is True

        # Wake word in middle - should match
        mock_rec.FinalResult.return_value = '{"text": "hey robot turn on the lights"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is True

        # Wake word at end - should match
        mock_rec.FinalResult.return_value = '{"text": "turn on the lights robot"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is True

        # No wake word - should not match
        mock_rec.FinalResult.return_value = '{"text": "hello there how are you"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is False

        # Empty text - should NOT match
        mock_rec.FinalResult.return_value = '{"text": ""}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is False


def test_check_for_wake_word_handles_multiple_wake_words():
    """Test check_for_wake_word works with multiple wake words"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot", "computer", "hey assistant"]
    wake_word_filter.fuzzy_matcher = None  # Exact matching
    wake_word_filter.use_grammar = True
    wake_word_filter.logger = Mock()

    with patch("src.models.wake_word_filter.KaldiRecognizer") as mock_recognizer_class:
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True
        mock_recognizer_class.return_value = mock_rec

        # First wake word at start
        mock_rec.FinalResult.return_value = '{"text": "robot do something"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is True

        # Second wake word at start
        mock_rec.FinalResult.return_value = '{"text": "computer show me"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is True

        # Multi-word wake word at start
        mock_rec.FinalResult.return_value = '{"text": "hey assistant what time"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is True


def test_check_for_wake_word_respects_word_boundaries():
    """Test check_for_wake_word doesn't match substrings"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot"]
    wake_word_filter.fuzzy_matcher = None
    wake_word_filter.use_grammar = True
    wake_word_filter.logger = Mock()

    with patch("src.models.wake_word_filter.KaldiRecognizer") as mock_recognizer_class:
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True
        mock_recognizer_class.return_value = mock_rec

        # Should match: exact word
        mock_rec.FinalResult.return_value = '{"text": "robot turn on"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is True

        # Should not match: wake word is part of another word
        mock_rec.FinalResult.return_value = '{"text": "robotics is cool"}'
        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is False


def test_check_for_wake_word_uses_grammar_when_enabled():
    """Test _check_for_wake_word uses grammar and returns True immediately on any text"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot", "computer"]
    wake_word_filter.fuzzy_matcher = None
    wake_word_filter.use_grammar = True
    wake_word_filter.logger = Mock()

    with patch("src.models.wake_word_filter.KaldiRecognizer") as mock_recognizer_class:
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True
        mock_rec.FinalResult.return_value = '{"text": "robot"}'
        mock_recognizer_class.return_value = mock_rec

        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )

        # Verify KaldiRecognizer was called with 3 args (model, sample_rate, grammar)
        call_args = mock_recognizer_class.call_args
        assert len(call_args[0]) == 3
        assert call_args[0][1] == 16000
        assert call_args[0][2] == '["robot", "computer"]'
        # Grammar mode returns True immediately when text is returned
        assert result is True


def test_check_for_wake_word_no_grammar_when_disabled():
    """Test _check_for_wake_word uses full transcription and searches for wake word"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot"]
    wake_word_filter.fuzzy_matcher = None
    wake_word_filter.use_grammar = False
    wake_word_filter.logger = Mock()

    with patch("src.models.wake_word_filter.KaldiRecognizer") as mock_recognizer_class:
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True
        mock_rec.FinalResult.return_value = '{"text": "robot turn on the lights"}'
        mock_recognizer_class.return_value = mock_rec

        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )

        # Verify KaldiRecognizer was called with 2 args (model, sample_rate) - no grammar
        call_args = mock_recognizer_class.call_args
        assert len(call_args[0]) == 2
        assert call_args[0][1] == 16000
        # Full transcription mode searches for wake word in text
        assert result is True


def test_check_for_wake_word_no_grammar_no_match():
    """Test _check_for_wake_word returns False when wake word not in transcription"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot"]
    wake_word_filter.fuzzy_matcher = None
    wake_word_filter.use_grammar = False
    wake_word_filter.logger = Mock()

    with patch("src.models.wake_word_filter.KaldiRecognizer") as mock_recognizer_class:
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True
        mock_rec.FinalResult.return_value = '{"text": "hello how are you"}'
        mock_recognizer_class.return_value = mock_rec

        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )

        assert result is False


def test_check_for_wake_word_handles_vosk_errors():
    """Test check_for_wake_word returns False on Vosk errors"""
    wake_word_filter = Mock()
    wake_word_filter.vosk_model = Mock()
    wake_word_filter.wake_words = ["robot"]
    wake_word_filter.fuzzy_matcher = None
    wake_word_filter.use_grammar = True
    wake_word_filter.logger = Mock()

    with patch("src.models.wake_word_filter.KaldiRecognizer") as mock_recognizer_class:
        mock_recognizer_class.side_effect = Exception("Vosk error")

        result = WakeWordFilter._check_for_wake_word(
            wake_word_filter, b"\x00" * 1000, 16000
        )
        assert result is False
        wake_word_filter.logger.error.assert_called_once()


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
    mic.get_properties.return_value = Mock(sample_rate_hz=48000, num_channels=1)
    wake_word_filter.microphone_client = mic

    with pytest.raises(ValueError, match="16000 Hz"):
        stream = await WakeWordFilter.get_audio(wake_word_filter, "pcm16", 0, 0)
        # Consume the generator to trigger the validation
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_get_properties_returns_mic_properties():
    """Test get_properties returns properties from underlying microphone"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    # Mock microphone with specific properties
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(sample_rate_hz=16000, num_channels=1)
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
    mic.get_properties.return_value = Mock(sample_rate_hz=48000, num_channels=2)
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
    mic.get_properties.return_value = Mock(sample_rate_hz=16000, num_channels=2)
    wake_word_filter.microphone_client = mic

    with pytest.raises(ValueError, match="mono.*1 channel"):
        stream = await WakeWordFilter.get_audio(wake_word_filter, "pcm16", 0, 0)
        # Consume the generator to wake_word_filter the validation
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_close_sets_shutdown_flag():
    """Test close() sets is_shutting_down flag"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    wake_word_filter.executor = Mock()

    await WakeWordFilter.close(wake_word_filter)

    assert wake_word_filter.is_shutting_down is True
    wake_word_filter.executor.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_close_shuts_down_executor():
    """Test close() properly shuts down thread pool executor"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    mock_executor = Mock()
    wake_word_filter.executor = mock_executor

    await WakeWordFilter.close(wake_word_filter)

    mock_executor.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_process_speech_segment_skips_when_shutting_down():
    """Test _process_speech_segment returns early if shutting down"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = True
    wake_word_filter.logger = Mock()

    chunk_buffer = [Mock(), Mock()]
    byte_buffer = bytearray(b"\x00" * 1000)

    # Should yield nothing when shutting down
    chunks = []
    async for chunk in WakeWordFilter._process_speech_segment(
        wake_word_filter, chunk_buffer, byte_buffer
    ):
        chunks.append(chunk)

    assert len(chunks) == 0
    wake_word_filter.logger.debug.assert_called_with(
        "Skipping speech processing due to shutdown"
    )


@pytest.mark.asyncio
async def test_process_speech_segment_handles_executor_shutdown_error():
    """Test _process_speech_segment handles RuntimeError during shutdown gracefully"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    wake_word_filter.logger = Mock()
    wake_word_filter.executor = Mock()

    chunk_buffer = [Mock()]
    byte_buffer = bytearray(b"\x00" * 1000)

    # Mock run_in_executor to raise RuntimeError with "shutdown" in message
    async def mock_run_in_executor(*args):
        raise RuntimeError("cannot schedule new futures after shutdown")

    with patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor

        chunks = []
        async for chunk in WakeWordFilter._process_speech_segment(
            wake_word_filter, chunk_buffer, byte_buffer
        ):
            chunks.append(chunk)

        assert len(chunks) == 0
        wake_word_filter.logger.debug.assert_called_with(
            "Executor shutdown during processing, ignoring"
        )


@pytest.mark.asyncio
async def test_process_speech_segment_yields_chunks_on_wake_word():
    """Test _process_speech_segment yields chunks when wake word detected"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    wake_word_filter.logger = Mock()
    wake_word_filter.executor = Mock()

    mock_chunk1 = Mock()
    mock_chunk2 = Mock()
    chunk_buffer = [mock_chunk1, mock_chunk2]
    byte_buffer = bytearray(b"\x00" * 1000)

    # Mock the wake word check to return True
    async def mock_run_in_executor(executor, func, *args):
        return True  # Wake word detected

    with patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor
        with patch("src.models.wake_word_filter.AudioChunk") as mock_audio_chunk_class:
            empty_chunk = Mock()
            empty_chunk.audio = Mock()
            mock_audio_chunk_class.return_value = empty_chunk

            chunks = []
            async for chunk in WakeWordFilter._process_speech_segment(
                wake_word_filter, chunk_buffer, byte_buffer
            ):
                chunks.append(chunk)

            # Should yield 2 audio chunks + 1 empty chunk at end
            assert len(chunks) == 3
            assert chunks[0] == mock_chunk1
            assert chunks[1] == mock_chunk2
            assert chunks[2] == empty_chunk


@pytest.mark.asyncio
async def test_process_speech_segment_yields_empty_chunk_at_end():
    """Test _process_speech_segment yields an empty AudioChunk at the end to signal segment end"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    wake_word_filter.logger = Mock()
    wake_word_filter.executor = Mock()

    mock_chunk = Mock()
    mock_chunk.audio = Mock()
    mock_chunk.audio.audio_data = b"\x00" * 100
    chunk_buffer = [mock_chunk]
    byte_buffer = bytearray(b"\x00" * 1000)

    # Mock the wake word check to return True
    async def mock_run_in_executor(executor, func, *args):
        return True  # Wake word detected

    with patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor
        with patch("src.models.wake_word_filter.AudioChunk") as mock_audio_chunk_class:
            empty_chunk = Mock()
            empty_chunk.audio = Mock()
            mock_audio_chunk_class.return_value = empty_chunk

            chunks = []
            async for chunk in WakeWordFilter._process_speech_segment(
                wake_word_filter, chunk_buffer, byte_buffer
            ):
                chunks.append(chunk)

            # Last chunk should be the empty AudioChunk
            assert len(chunks) == 2
            assert chunks[0] == mock_chunk
            assert chunks[1] == empty_chunk


@pytest.mark.asyncio
async def test_process_speech_segment_yields_nothing_when_no_wake_word():
    """Test _process_speech_segment yields nothing when wake word not detected"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    wake_word_filter.logger = Mock()
    wake_word_filter.executor = Mock()

    chunk_buffer = [Mock(), Mock()]
    byte_buffer = bytearray(b"\x00" * 1000)

    # Mock the wake word check to return False
    async def mock_run_in_executor(executor, func, *args):
        return False  # No wake word

    with patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor

        chunks = []
        async for chunk in WakeWordFilter._process_speech_segment(
            wake_word_filter, chunk_buffer, byte_buffer
        ):
            chunks.append(chunk)

        assert len(chunks) == 0
