# Module filtered-audio

filtered-audio module provides a model to filter audio input from a source microphone based on
wake words.

## Supported Platforms
- **Darwin ARM64**
- **Linux x64**
- **Linux ARM64**

## Models

This module provides the following model(s):

- [`viam:filtered-audio:wake-word-filter`]



## Model viam:filtered-audio:wake-word-filter
### Configuration
The following attribute template can be used to configure this model:

```json
{
  "source_microphone" : <AUDIO_IN NAME>,
  "wake_words": ["<word>"]
}
```

``
#### Configuration Attributes

The following attributes are available for the `viam:audio:microphone` model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `source_microphone` | string | **Required** | Name of a Viam AudioIn component to recieve and filter audio from
| `wake_words` | string array | **Required** | Wake words to filter speech. All speech segments said after the wake words will be returned from get_audio
| `vosk_model` | string | **Optional** | The name of the VOSK model to use for speech to text. Default: vosk-model-small-en-us-0.15.
| `vad_aggressiveness` | int | **Optional** | Sensitivity of the webRTC VAD (voice activity detection). A higher number is more restrictive in reporting speech, and missed detection rates go up. A lower number is less restrictive but may report background noise. Range: 0-3. Default: 3.

### Source Microphone Requirements

The source microphone **must** provide audio in the following format:

| Requirement | Value | Description |
|-------------|-------|-------------|
| **Codec** | PCM16 | 16-bit PCM audio format |
| **Sample Rate** | 16000 Hz | Required for the default Vosk model |
| **Channels** | 1 (Mono) | Stereo audio is not supported |

**Example configuration for source microphone:**
```json
{
  "name": "my-microphone",
  "type": "audio_in",
  "model": "...",
  "attributes": {
    "sample_rate": 16000,
    "channels": 1
  }
}
```

**Note:** The wake-word-filter validates these requirements on startup and raises an error if the source microphone doesn't meet them.

**Recommended Source Microphone:** Use the [`viam:system-audio`](https://app.viam.com/module/viam/system-audio) module, which supports resampling and can output 16 kHz mono PCM16 audio from any system microphone.
