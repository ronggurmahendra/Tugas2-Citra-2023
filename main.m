function main()
    clc;
    close all;
    clear all;
    my_conv_test()
end
% spek 1

function result = my_conv(signal, kernel)
    % Get the lengths of the input sequences
    signal_len = length(signal);
    kernel_len = length(kernel);
    
    % Initialize the result vector
    result = zeros(1, signal_len + kernel_len - 1);

    % Perform convolution
    for n = 1:length(result)
        result(n) = 0;
        for k = 1:kernel_len
            if n - k + 1 > 0 && n - k + 1 <= signal_len
                result(n) = result(n) + kernel(k) * signal(n - k + 1);
            end
        end
    end
end
% Spek 2
% Tulislah program Matlab untuk melakukan image smoothing (pada citra yang mengandung derau)
% atau blurring pada citra dalam:
% - ranah spasial menggunakan mean filter n x n dan gaussian filter n x n. Gunakan fungsi
% konvolusi yang telah anda kerjakan pada poin 1.
% - Ranah frekuensi menggunakan low-pass filter ILPF, GLPF, dan BLPF.
% Citra yang digunakan adalah citra grayscale dan citra berwarna. Uji program dengan
% menggunakan tiga contoh citra grayscale dan tiga contoh citra berwarna berikut dan dua citra
% tambahan (grayscale dan berwarna) yang Anda cari sendiri.
Mask1 = [1/16,2/16,1/16 ; 2/16,4/16,2/16 ; 1/16,2/16,1/16];
Mask2 = [0,-1,0 ; -1,-4,-1 ; 0,-1,0];
% Gaussian Mask
Mask3 = [] ;
Mask4 = [ -1,-1,-1 ; -1,-17,-1 ; -1,-1,-1];

function my_conv_test()

end


% Baca citra grayscale
grayscale_image = imread('grayscale_image.png');
% Konversi ke domain frekuensi
grayscale_image_fft = fftshift(fft2(grayscale_image));

% Baca citra berwarna
color_image = imread('color_image.png');
% Konversi ke domain frekuensi
color_image_fft = fftshift(fft2(rgb2gray(color_image)));

% Parameter filter high-pass
D0 = 50; % Parameter jarak cutoff
n = 2;   % Orde filter Butterworth

% Filter IHPF
IHPF = ones(size(grayscale_image_fft));
[m, n] = size(IHPF);
center_x = round(m / 2);
center_y = round(n / 2);
radius = D0;
for i = 1:m
    for j = 1:n
        distance = sqrt((i - center_x)^2 + (j - center_y)^2);
        if distance <= radius
            IHPF(i, j) = 0;
        end
    end
end

% Filter GHPF
GHPF = 1 - exp(-(grayscale_image_fft.^2) / (2 * D0^2));

% Filter BHPF
BHPF = 1 ./ (1 + (D0 ./ distance_matrix(grayscale_image)).^(2 * n));

% Terapkan filter pada citra grayscale
filtered_grayscale_IHPF = ifft2(ifftshift(grayscale_image_fft .* IHPF));
filtered_grayscale_GHPF = ifft2(ifftshift(grayscale_image_fft .* GHPF));
filtered_grayscale_BHPF = ifft2(ifftshift(grayscale_image_fft .* BHPF));

% Tampilkan citra hasil
imshow(filtered_grayscale_IHPF, []);
title('Filtered Grayscale Image - IHPF');
imshow(filtered_grayscale_GHPF, []);
title('Filtered Grayscale Image - GHPF');
imshow(filtered_grayscale_BHPF, []);
title('Filtered Grayscale Image - BHPF');

% Terapkan filter pada citra berwarna
filtered_color_IHPF = ifft2(ifftshift(color_image_fft .* IHPF));
filtered_color_GHPF = ifft2(ifftshift(color_image_fft .* GHPF));
filtered_color_BHPF = ifft2(ifftshift(color_image_fft .* BHPF));

% Tampilkan citra hasil
figure;
imshow(uint8(filtered_color_IHPF));
title('Filtered Color Image - IHPF');
figure;
imshow(uint8(filtered_color_GHPF));
title('Filtered Color Image - GHPF');
figure;
imshow(uint8(filtered_color_BHPF));
title('Filtered Color Image - BHPF');
