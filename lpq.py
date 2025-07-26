import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist


def lpq(img, winSize=7, decorr=1, mode="nh"):
    rho = 0.90

    STFTalpha = (
        1 / winSize
    )  # alpha in STFT approaches (for Gaussian derivative alpha=1)

    convmode = "valid"  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window

    #  STFT uniform window
    #  Basic STFT filters
    w0 = np.ones_like(x)
    w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
    w2 = np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack(
        [
            filterResp1.real,
            filterResp1.imag,
            filterResp2.real,
            filterResp2.imag,
            filterResp3.real,
            filterResp3.imag,
            filterResp4.real,
            filterResp4.imag,
        ]
    )

    if decorr == 1:
        xp, yp = np.meshgrid(np.arange(1, winSize + 1), np.arange(1, winSize + 1))
        pp = np.column_stack((yp.flatten(), xp.flatten()))
        dd = cdist(pp, pp)
        C = rho**dd

        q1 = w0.reshape((winSize, 1)) @ w1.reshape((1, winSize))
        q2 = w1.reshape((winSize, 1)) @ w0.reshape((1, winSize))
        q3 = w1.reshape((winSize, 1)) @ w1.reshape((1, winSize))
        q4 = w1.reshape((winSize, 1)) @ w2.reshape((1, winSize))

        M = np.vstack(
            (
                q1.real.T.ravel(),
                q1.imag.T.ravel(),
                q2.real.T.ravel(),
                q2.imag.T.ravel(),
                q3.real.T.ravel(),
                q3.imag.T.ravel(),
                q4.real.T.ravel(),
                q4.imag.T.ravel(),
            )
        )

        D = np.dot(M, C).dot(M.T)
        A = np.diag(
            [1.000007, 1.000006, 1.000005, 1.000004, 1.000003, 1.000002, 1.000001, 1]
        )
        U, S, V = np.linalg.svd(np.dot(A, D).dot(A))
        V = V.T

        freqRespShape = freqResp.shape
        freqResp = freqResp.reshape((-1, freqResp.shape[2]))
        freqResp = np.dot(V.T, freqResp.T).T
        freqResp = freqResp.reshape(freqRespShape)
        freqRespDecorr = freqResp.copy()

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode == "im":
        LPQdesc = np.uint8(LPQdesc)

    ## Histogram if needed
    if mode == "nh" or mode == "h":
        LPQdesc = np.histogram(LPQdesc.flatten(), range(257))[0]

    ## Normalize histogram if needed
    if mode == "nh":
        LPQdesc = LPQdesc / LPQdesc.sum()

    return LPQdesc


def extract_lpq_features_for_each_category(
    categories, images, winSize=7, decorr=1, mode="nh"
):
    return {
        category: [lpq(img, winSize, decorr, mode) for img in images[category]]
        for category in categories
    }
