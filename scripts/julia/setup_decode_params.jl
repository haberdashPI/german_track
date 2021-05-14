using GermanTrack
using EEGCoding

params = let
    samplerate = 32
    max_lag = 3
    nlags = round(Int,samplerate*max_lag)
    lags = -(nlags-1):1:0
    decode_sr = 1 / (round(Int, 0.1samplerate) / samplerate)
    nλ = 6

    params = (
        stimulus = (
            samplerate = samplerate,
            max_lag = max_lag,
            nlags = nlags,
            lags = lags,
            encodings = Dict("pitch" => PitchSurpriseEncoding(), "envelope" => ASEnvelope()),

            sources = [
                male_source,
                fem1_source,
                fem2_source,
                # male_fem1_sources,
                # male_fem2_sources,
                # fem1_fem2_sources
            ],
        ),

        train = (
            nfolds = 5,
            max_steps = 50,
            min_steps = 6,
            hidden_units = 64,
            patience = 6,
            # while testing new pipelines, we use an decently good λ
            # (this is just hand picked based on earlier runs)
            # λs = [0.016],
            # utlimately, on a final run, we run a gamut of λs to pick the best one
            # by cross-validation
            λs = exp.(range(log(1e-3), log(1e-1), length = nλ)),
            batchsize = 1024,
        ),

        test = (
            decode_sr = decode_sr,
            winlen_s = 1.0,
        ),
    )
end
