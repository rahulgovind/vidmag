By default, the scripts are written so as to
all result files automatically.

## Audio reconstruction results
| Title | Original Video Link | Original Audio | Reconstructed Audio | Reconstructed denoised
| ------|---------------------|----------------|---------------------|-----------------------|
|Chips2 - Mary had a little lamb  tune|[Original AVI 10.8 GB](http://data.csail.mit.edu/vidmag/VisualMic/Results/Chips2-2200Hz-Mary_MIDI-input.avi) | [chips2-mary-original.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/ground-truth-audio/Chips2-2200Hz-Mary_MIDI-input.wav?raw=True) |[chips2-mary.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/reconstructed-audio/chips2-mary.wav?raw=True) | [chips2-mary-denoised.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/reconstructed-audio/chips2-mary-denoised.wav?raw=True) |
|Plant  - Mary had a little lamb tune| [Original AVI 12.1 GB](http://data.csail.mit.edu/vidmag/VisualMic/Results/Plant-2200Hz-Mary_MIDI-input.avi) | [plant-mary-original.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/ground-truth-audio/Chips2-2200Hz-Mary_MIDI-input.wav?raw=True) | [plant-mary.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/reconstructed-audio/plant-mary.wav?raw=True) | [plant-mary-denoised.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/reconstructed-audio/plant-mary-denoised.wav?raw=True)
|Chips1 - Human speech | [Original AVI 13.3 GB](http://data.csail.mit.edu/vidmag/VisualMic/Results/Chips1-2200Hz-Mary_Had-input.avi) | [chips1-mary-mic.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/ground-truth-audio/Chips1-2200Hz-Mary_Had-input_resampled_to_video_rate.wav?raw=True) | [chips1-mary-reconstructed.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/reconstructed-audio/chips2-mary.wav?raw=True) | [chips1-mary-denoised.wav](https://github.com/rahulgovind/vidmag/blob/towards_audio/reconstructed-audio/plant-mary-denoised.wav?raw=True)

## Video magnification results

| Title | Original Video Link | Amplification method | Amplified Video |
| ------|---------------------|--------------------|-----------------|
| Baby breathing | [baby.mp4](http://people.csail.mit.edu/mrub/evm/video/baby.mp4) | Linear (Eulerian Motion Magnification Wu et al) |
| Baby breathing | [baby.mp4](http://people.csail.mit.edu/mrub/evm/video/baby.mp4) | Phase-based (Using Riesz pyramids) | [Amplified Video](https://github.com/rahulgovind/vidmag/blob/towards_audio/amplified-videos/phase-based/baby.mp4?raw=true)
