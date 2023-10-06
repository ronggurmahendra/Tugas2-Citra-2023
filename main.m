function main()
    clc;
    close all;
    clear all;
    % my_conv_test()
    % my_smoothing_test()
    spek3()
end
% spek 1

function result = my_grey_conv(image, mask)
    [m, n] = size(image);
    [p, q] = size(mask);
    pad = floor((p - 1) / 2); % padding
    result = zeros(m, n);
    imagePadded = padarray(image, [pad, pad], 0, 'both');
    % Melakukan konvolusi
    for i = 1:m
        for j = 1:n
            % Mendapatkan bagian dari citra yang sesuai dengan ukuran mask
            region = imagePadded(i:i+p-1, j:j+q-1);

            % Melakukan konvolusi
            result(i, j) = sum(sum(region .* mask));
        end
    end
end

function result = my_rgb_conv(image, mask)
    [m, n, ~] = size(image);  % Get the dimensions and ignore the third dimension (color channels)
    [p, q] = size(mask);
    pad = floor((p - 1) / 2); % Padding
    result = zeros(m, n, 3);  % Initialize result for three color channels
    imagePadded = padarray(image, [pad, pad], 0, 'both');

    for c = 1:3 
        for i = 1:m
            for j = 1:n
                region = imagePadded(i:i+p-1, j:j+q-1, c);

                % Melakukan konvolusi
                result(i, j, c) = sum(sum(region .* mask));
            end
        end
    end
end



function my_conv_test()
    masktest = [1 2 1; 0 0 0; -1 -2 -1]
    % mask1 = [1/16 2/16 1/16 ; 2/16 4/16 2/16 ; 1/16 2/16 1/16];
    mask1 = [1 2 1 ; 2 4 2 ; 1 2 1]/16;
    mask2 = [0 -1 0 ; -1 -4 -1 ; 0 -1 0];
    mask3 = [1 1 2 2 2 1 1 ; 1 2 2 4 2 2 1 ; 2 2 4 8 4 2 2 ; 2 4 8 16 8 4 2 ; 2 2 4 8 4 2 2 ; 1 2 2 4 2 2 1 ; 1 1 2 2 2 1 1] / 140 ;
    mask4 = [ -1 -1 -1 ; -1 17 -1 ; -1 -1 -1];
    img_input = imread("1.jpg");   
    % imageGrayscale = rgb2gray(img_input);
    % print(mask2)
    figure;
    
    % imshow(imageGrayscale)
    % img_result = my_grey_conv(double(imageGrayscale), mask1);        
    img_result = uint8(my_grey_conv(double(img_input), mask3));
    figure;
    imshow(img_result)

    hasilKonvolusiMatlab = uint8(convn (double(img_input), mask3, 'same'));
    figure;
    imshow(hasilKonvolusiMatlab)
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

function result = image_mean_grey(image,n)
    mean_mask = ones(n) / (n * n);
    result = my_grey_conv(image, mean_mask);
end


function result = image_mean_rgb(image,n)
    mean_mask = ones(n) / (n * n);
    result = my_rgb_conv(image, mean_mask);
end


function result = image_gauss_grey(image,n)
    sigma = 2.0; % Change the sigma value as needed
    gaussian_mask = fspecial('gaussian', [n n], sigma);
    result = my_grey_conv(image, gaussian_mask);
end


function result = image_gauss_rgb(image,n, sigma)
    % sigma = 2.0; % Change the sigma value as needed
    gaussian_mask = fspecial('gaussian', [n n], sigma);
    result = my_rgb_conv(image, gaussian_mask);
end


function my_smoothing_test()
    filename = "1.jpg"
    % filename = 'image-011.jpg';
    img_input = imread(filename);   
    % imageGrayscale = rgb2gray(img_input);
    % print(mask2)
    figure;
    imshow(img_input)
    % disp(length("1.jpg"))
    % isGreyscale = endsWith(lower(filename), '.bmp');

    % % Extract the last 3 characters
    % lastThreeChars = filename(length(filename)-2:length(filename));
    
    % % Check if it's "jpg" (case insensitive)
    % isGreyscale = strcmpi(lastThreeChars, 'bmp');

    % if isGreyscale
    %     img_result = uint8(image_mean_grey(double(img_input), 3));
    % else
    %     img_result = uint8(image_mean_rgb(double(img_input), 3));
    % end
    % img_result = uint8(image_mean_rgb(double(img_input), 11));
    img_result = uint8(image_gauss_rgb(double(img_input), 11, 2));
    
    figure;
    imshow(img_result)
end

% Spek3
% Tulislah program Matlab untuk melakukan penapisan citra dalam ranah frekuensi menggunakan
% high-pass filter IHPF, GHPF, dan BHPF. Citra yang digunakan adalah citra grayscale dan citra
% berwarna.
% Uji program dengan menggunakan contoh citra berikut dan dua citra tambahan yang anda cari
% % sendiri
% function spek3()
%     % Baca citra grayscale
%     filename = "1.bmp"
%     grayscale_image = imread(filename);
%     % Konversi ke domain frekuensi
%     grayscale_image_fft = fftshift(fft2(grayscale_image));

%     % Baca citra berwarna
%     filename = "1.jpg"
%     color_image = imread(filename);
%     % Konversi ke domain frekuensi
%     color_image_fft = fftshift(fft2(rgb2gray(color_image)));

%     % Parameter filter high-pass
%     D0 = 50; % Parameter jarak cutoff
%     n = 2;   % Orde filter Butterworth

%     % Filter IHPF
%     IHPF = ones(size(grayscale_image_fft));
%     [m, n] = size(IHPF);
%     center_x = round(m / 2);
%     center_y = round(n / 2);
%     radius = D0;
%     for i = 1:m
%         for j = 1:n
%             distance = sqrt((i - center_x)^2 + (j - center_y)^2);
%             if distance <= radius
%                 IHPF(i, j) = 0;
%             end
%         end
%     end

%     % Filter GHPF
%     GHPF = 1 - exp(-(grayscale_image_fft.^2) / (2 * D0^2));

%     % Filter BHPF
%     % BHPF = 1 ./ (1 + (D0 ./ distance_matrix(grayscale_image)).^(2 * n));

%     % Terapkan filter pada citra grayscale
%     filtered_grayscale_IHPF = ifft2(ifftshift(grayscale_image_fft .* IHPF));
%     filtered_grayscale_GHPF = ifft2(ifftshift(grayscale_image_fft .* GHPF));
%     % filtered_grayscale_BHPF = ifft2(ifftshift(grayscale_image_fft .* BHPF));

%     % Tampilkan citra hasil
%     imshow(filtered_grayscale_IHPF, []);
%     title('Filtered Grayscale Image - IHPF');
%     imshow(filtered_grayscale_GHPF, []);
%     title('Filtered Grayscale Image - GHPF');
%     % imshow(filtered_grayscale_BHPF, []);
%     % title('Filtered Grayscale Image - BHPF');

%     % Terapkan filter pada citra berwarna
%     filtered_color_IHPF = ifft2(ifftshift(color_image_fft .* IHPF));
%     filtered_color_GHPF = ifft2(ifftshift(color_image_fft .* GHPF));
%     % filtered_color_BHPF = ifft2(ifftshift(color_image_fft .* BHPF));

%     % Tampilkan citra hasil
%     figure;
%     imshow(uint8(filtered_color_IHPF));
%     title('Filtered Color Image - IHPF');
%     figure;
%     imshow(uint8(filtered_color_GHPF));
%     title('Filtered Color Image - GHPF');
%     % figure;
%     % imshow(uint8(filtered_color_BHPF));
%     % title('Filtered Color Image - BHPF');
% end

function spek3()
    % Baca citra grayscale
    gray_image = imread('1.jpg');

    % Baca citra berwarna
    color_image = imread('1.jpg');

    % Tentukan ukuran filter
    filter_size = 31;

    % Buat filter IHPF
    ihpf_filter = fspecial('unsharp', 0.5);

    % Buat filter GHPF
    ghpf_filter = fspecial('gaussian', filter_size, 5);

    % Buat filter BHPF
    % bhp_filter = fspecial('butter', 2, 0.5);

    % Terapkan filter pada citra grayscale
    filtered_gray_ihpf = imfilter(gray_image, ihpf_filter);
    filtered_gray_ghpf = imfilter(gray_image, ghpf_filter);
    % filtered_gray_bhp = imfilter(gray_image, bhp_filter);

    % Terapkan filter pada citra berwarna
    filtered_color_ihpf = imfilter(color_image, ihpf_filter);
    filtered_color_ghpf = imfilter(color_image, ghpf_filter);
    % filtered_color_bhp = imfilter(color_image, bhp_filter);

    % Tampilkan citra asli dan citra hasil penapisan
    figure;

    subplot(3, 2, 1);
    imshow(gray_image);
    title('Citra Grayscale Asli');

    subplot(3, 2, 2);
    imshow(filtered_gray_ihpf);
    title('Citra Grayscale IHPF');

    subplot(3, 2, 3);
    imshow(filtered_gray_ghpf);
    title('Citra Grayscale GHPF');

    % subplot(3, 2, 4);
    % imshow(filtered_gray_bhp);
    % title('Citra Grayscale BHPF');

    subplot(3, 2, 5);
    imshow(color_image);
    title('Citra Berwarna Asli');

    subplot(3, 2, 6);
    imshow(filtered_color_ihpf);
    title('Citra Berwarna IHPF');

    % Tampilkan citra hasil penapisan GHPF dan BHPF
    figure;

    subplot(1, 2, 1);
    imshow(filtered_color_ghpf);
    title('Citra Berwarna GHPF');

    % subplot(1, 2, 2);
    % imshow(filtered_color_bhp);
    % title('Citra Berwarna BHPF');

end

% Spek 4
% Tulislah program Matlab untuk melakukan penapisan citra dalam ranah frekuensi untuk
% menghasilkan citra yang lebih terang seperti di bawah ini. Pikirkan bagaimana car

% Spek 5 
% Buatlah program Matlab untuk menambahkan derau salt and pepper pada citra (grayscale dan
% berwarna), lalu lakukan penghilangan derau dengan min fiter, max filter, median filter, arithmetic
% mean filter, geometric filter, harmonic mean filter, contraharmonic mean filter, midpoint filter,
% alpha-trimmed mean filter. Tidak boleh menggunakan fungsi medfilt2 di dalam Matlab. Uji
% program dengan menggunakan contoh citra berikut dan dua citra tambahan yang anda cari
% sendiri.

% Spek 6
% Pikirkan bagaimana cara menghilangan derau periodik pada citra berikut, lalu tulislah program
% Matlab nya.

% Spek 7
% Buatlah program yang melakukan motion bluring pada citra (grayscale dan berwarna),
% lalu lakukan dekonvolusi pada citra tersebut dengan penapis Wiener. Program penapis
% Wiener anda buat sendiri (tidak boleh menggunakan fungsi Wiener di dalam Matlab).
% Ujicoba pada dua citra di bawah ini dan dua citra tambahan:

% mas royan drone
% mas ardimas research
% mas renda finance

% pertanyaan : 
% bobot kerjanya relatif ya 7 jam
% senin kebetulan ada ujian pagi - pagi itu onboarding mulai jamber ya 

% 50% kerja 50% bljr


% No rek 