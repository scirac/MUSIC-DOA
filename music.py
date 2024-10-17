import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks

# Parameters
DOA = [-10,10]      # Direction of arrival (Degree)
T = 512            # Snapshots (or Samples)
K = len(DOA)       # The number of signal source(or target)
Nr = 12           # Number of receiver's antennas
lambda_ = 0.343      # Wavelength
d = lambda_ / 2   # Receiver's antennas spacing
SNR = 1            # Signal to Noise Ratio (dB)

# Steering Matrix 每一列对应一个入射方向的导向矢量
A = np.zeros((Nr, K), dtype=complex)
for k in range(K):
    A[:, k] = np.exp(-1j * 2 * np.pi * d * np.sin(np.radians(DOA[k])) * np.arange(Nr) / lambda_)

Vj = np.sqrt((10 ** (SNR / 10)) / 2)
s = Vj * (np.random.randn(K, T) + 1j * np.random.randn(K, T))
noise = np.sqrt(1 / 2) * (np.random.randn(Nr, T) + 1j * np.random.randn(Nr, T))
X = A @ s
X = X + noise  # Insert Additive White Gaussian Noise (AWGN)

# MUSIC (MUltiple SIgnal Classification)
Rx = np.cov(X)  # Data covariance matrix
print(Rx.shape)
eigenVal, eigenVec = eigh(Rx)  # Find the eigenvalues and eigenvectors of Rx
Vn = eigenVec[:, :Nr - K]  # Estimate noise subspace (Note that eigenvalues sorted ascending on columns of "eigenVal")

theta = np.arange(-90, 90, 0.5)  # Grid points of Peak Search
Pmusic = []
for angle in theta:
    SS = np.exp(-1j * 2 * np.pi * d * np.arange(Nr) * np.sin(np.radians(angle)) / lambda_)
    PP = SS.conj().T @ (Vn @ Vn.conj().T) @ SS
    Pmusic.append(1 / PP)

Pmusic = np.real(10 * np.log10(Pmusic))  # Spatial Spectrum function
pks, properties = find_peaks(Pmusic, height=None, distance=20) #pks 返回峰值对应的索引值
peak_values = Pmusic[pks]  #保存峰值
locs = theta[pks]  # 保存峰值对应角度
peak_sorted_index = np.argsort(peak_values)[::-1] #对峰值从大到小排序并返回对应索引
MUSIC_Estim = (locs[peak_sorted_index])[:K]


# Plotting
plt.figure()
plt.plot(theta, Pmusic, '-b')
plt.plot(MUSIC_Estim, Pmusic[np.isin(theta, MUSIC_Estim)], 'r*')
for x, y in zip(MUSIC_Estim, Pmusic[np.isin(theta, MUSIC_Estim)]):
    plt.text(x + 2 * np.sign(x), y, f'{x:.2f}')
plt.xlabel('Angle θ (degree)')
plt.ylabel('Spatial Power Spectrum P(θ) (dB)')
plt.title('DOA estimation based on MUSIC algorithm')
plt.xlim([min(theta), max(theta)])
plt.grid(True)
plt.show()

print("Estimated DOAs:", MUSIC_Estim)
