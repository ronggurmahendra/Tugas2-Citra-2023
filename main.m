function main()
    clc;
    close all;
    clear all;
    % my_conv_test()
    % my_smoothing_test()
    % spek3()
    % spek4()
    % spek5()
    spek6()
    % spek7()
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

    [M, N] = size(color_image); 
    % tranform fourier
    FT_img = fft2(double(color_image)); 
    n = 2
    D0 = 10;

    u = 0:(M-1); 
    v = 0:(N-1); 
    idx = find(u > M/2); 
    u(idx) = u(idx) - M; 
    idy = find(v > N/2); 
    v(idy) = v(idy) - N; 
     
    [V, U] = meshgrid(v, u); 


    D = sqrt(U.^2 + V.^2); 

    H = 1./(1 + (D0./D).^(2*n)); 

    G = H.*FT_img; 
    %Inverse fouriser transform
    img_result = real(ifft2(double(G)));  


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
function spek4()
    % Baca citra grayscale
    gray_image = imread('image-017.bmp');
    
    % Baca citra berwarna
    color_image = imread('image-019.jpg');
    h_size = 10
    Sigma = 10 
    A = 3
    GLPF = fspecial('gaussian',h_size,Sigma);
    GLPF_Image = imfilter(color_image,GLPF);
    
    High_pass_filter = color_image - GLPF_Image;

    Unsharpened_Image = (1*color_image) + High_pass_filter;

    HFBF_Image = (A-1)*color_image + color_image - GLPF_Image;
    figure(),imshow(color_image),
    title('Original Image')
    figure(),imshow(GLPF_Image);
    title('Gaussian Low Pass Filtered Image');
    figure(),imshow(Unsharpened_Image);
    title('Unsharpened Image');
    figure(),imshow(HFBF_Image);
    title('High Frequency Boosted Image');
    
    % figure;
    % subplot(1, 2, 1);
    % imshow(gray_image);
    % title('Citra Asli');

    % subplot(1, 2, 2);
    % imshow(filteredImage);
    % title('Citra Terang');
end

% Spek 5 
% Buatlah program Matlab untuk menambahkan derau salt and pepper pada citra (grayscale dan
% berwarna), lalu lakukan penghilangan derau dengan min fiter, max filter, median filter, arithmetic
% mean filter, geometric filter, harmonic mean filter, contraharmonic mean filter, midpoint filter,
% alpha-trimmed mean filter. Tidak boleh menggunakan fungsi medfilt2 di dalam Matlab. Uji
% program dengan menggunakan contoh citra berikut dan dua citra tambahan yang anda cari
% sendiri.
function outputImage = min_filter(inputImage, kernelSize)
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                          max(1, j - radius):min(cols, j + radius), c);
                
                outputImage(i, j, c) = min(neighborhood(:));
            end
        end
    end
end

function outputImage = max_filter(inputImage, kernelSize)
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                
                outputImage(i, j,c) = max(neighborhood(:));
            end
        end
    end
end
function outputImage = median_filter(inputImage, kernelSize)

    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                % Get the neighborhood of the current pixel
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                
                % Apply the median filter operation
                outputImage(i, j,c) = median(neighborhood(:));
            end
        end
    end
end

function outputImage = arithmetic_mean_filter(inputImage, kernelSize)
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                
                outputImage(i, j,c) = mean(neighborhood(:));
            end
        end
    end
end

function outputImage = geometric_filter(inputImage, kernelSize, angle_degrees)

    
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    
    % Convert the angle from degrees to radians
    angle_radians = deg2rad(angle_degrees);
    
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols

                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                

                rotated_neighborhood = imrotate(neighborhood, angle_degrees, 'bilinear', 'crop');
                
                center = floor(size(rotated_neighborhood) / 2) + 1;
                
                outputImage(i, j,c) = rotated_neighborhood(center(1), center(2));
            end
        end
    end
end
function outputImage = harmonic_mean_filter(inputImage, kernelSize)

    
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                % Get the neighborhood of the current pixel
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                
                reciprocal_neighborhood = 1 ./ double(neighborhood);
                harmonic_mean = kernelSize^2 / sum(reciprocal_neighborhood(:));
                
                outputImage(i, j,c) = harmonic_mean;
            end
        end
    end
end
function outputImage = contraharmonic_mean_filter(inputImage, kernelSize, q)

    
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                
                numerator = sum(neighborhood(:) .^ (q + 1));
                denominator = sum(neighborhood(:) .^ q);
                
                if denominator == 0
                    outputImage(i, j,c) = 0;
                else
                    outputImage(i, j,c) = numerator / denominator;
                end
            end
        end
    end
end
function outputImage = midpoint_filter(inputImage, kernelSize)

    
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                
                outputImage(i, j,c) = median(neighborhood(:));
            end
        end
    end
end
function outputImage = alpha_trimmed_mean_filter(inputImage, kernelSize, d)

    
    % dimensi gambar
    [rows, cols, channels] = size(inputImage);
    
    % init output
    outputImage = zeros(rows, cols, channels);
    
    
    % radius kernel
    radius = (kernelSize - 1) / 2;
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                neighborhood = inputImage(max(1, i - radius):min(rows, i + radius), ...
                                        max(1, j - radius):min(cols, j + radius),c);
                
                sorted_neighborhood = sort(neighborhood(:));
                
                trimmed_neighborhood = sorted_neighborhood((d/2 + 1):(end - d/2));
                
                outputImage(i, j,c) = mean(trimmed_neighborhood);
            end
        end
    end
end


function spek5()
    % Baca citra grayscale
    gray_image = imread('image-021.bmp');
    gray_myImage = imread('myImage.bmp');

    
    % Baca citra berwarna
    color_image = imread('image-022.jpg');
    color_myImage = imread('myImage.jpg');
    
    kernelSize = 3
    color_image_saltnpepper = imnoise( color_image ,'salt & pepper')

    min_denoised_image = min_filter(color_image_saltnpepper,kernelSize);
    max_denoised_image = max_filter(color_image_saltnpepper,kernelSize);
    median_denoised_image = median_filter(color_image_saltnpepper,kernelSize);
    arithmetic_mean_denoised_image = arithmetic_mean_filter(color_image_saltnpepper,kernelSize);
    geometric_denoised_image = geometric_filter(color_image_saltnpepper,kernelSize,45);
    harmonic_mean_denoised_image = harmonic_mean_filter(color_image_saltnpepper,kernelSize);
    % contraharmonic_mean_denoised_image = contraharmonic_mean_filter(color_image_saltnpepper,kernelSize, 1.5);
    midpoint_denoised_image = midpoint_filter(color_image_saltnpepper,kernelSize);
    alpha_trimmed_mean_denoised_image = alpha_trimmed_mean_filter(color_image_saltnpepper,kernelSize,2);

    figure;
    subplot(5, 2, 1);
    imshow(color_image);
    title('Original Asli');

    subplot(5, 2, 2);
    imshow(color_image_saltnpepper);
    title('Citra saltnpepper');

    subplot(5, 2, 3);
    imshow(uint8(min_denoised_image));
    title('Citra min_denoised_image');

    subplot(5, 2, 4);
    imshow(uint8(max_denoised_image));
    title('Citra max_denoised_image');

    subplot(5, 2, 5);
    imshow(uint8(median_denoised_image));
    title('Citra median_denoised_image');

    subplot(5, 2, 6);
    imshow(uint8(arithmetic_mean_denoised_image));
    title('Citra arithmetic_mean_denoised_image');

    subplot(5, 2, 7);
    imshow(uint8(geometric_denoised_image));
    title('Citra geometric_denoised_image');

    subplot(5, 2, 8);
    imshow(uint8(harmonic_mean_denoised_image));
    title('Citra harmonic_mean_denoised_image');

    % subplot(5, 2, 9);
    % imshow(contraharmonic_mean_denoised_image);
    % title('Citra contraharmonic_mean_denoised_image');

    subplot(5, 2, 9);
    imshow(uint8(midpoint_denoised_image));
    title('Citra midpoint_denoised_image');
    subplot(5, 2, 10);
    imshow(uint8(alpha_trimmed_mean_denoised_image));
    title('Citra alpha_trimmed_mean_denoised_image');
end

% Spek 6
% Pikirkan bagaimana cara menghilangan derau periodik pada citra berikut, lalu tulislah program
% Matlab nya.
function spek6()
    % Baca citra grayscale
    grayImage = imread('image-021.bmp');
    gray_myImage = imread('myImage.bmp');

    
    % Baca citra berwarna
    color_image = imread('image-022.jpg');
    color_myImage = imread('myImage.jpg');
    
    frequencyImage = fftshift(fft2(grayImage));
    % Take log magnitude so we can see it better in the display.
    amplitudeImage = log(abs(frequencyImage));
    minValue = min(min(amplitudeImage))
    maxValue = max(max(amplitudeImage))
    subplot(2, 3, 4);
    imshow(amplitudeImage, []);
    % caption = sprintf('Notice the two spikes\nperpendicular to the periodic frequency');
    % title(caption, 'FontSize', fontSize);
    % axis on;
    % zoom(10)
    
    % Find the location of the big spikes.
    amplitudeThreshold = 10.9;
    brightSpikes = amplitudeImage > amplitudeThreshold; % Binary image.
    % subplot(2, 3, 5);
    % imshow(brightSpikes);
    % axis on;
    % title('Bright Spikes', 'FontSize', fontSize);
    % Let user see the image.
    promptMessage = sprintf('The image below shows the bright spikes.\nNow we will exclude the central spike.');
    titleBarCaption = 'Continue?';
    button = questdlg(promptMessage, titleBarCaption, 'Continue', 'Cancel', 'Continue');
    if strcmpi(button, 'Cancel')
        return;
    end
    % Exclude the central DC spike.  Everything from row 115 to 143.
    brightSpikes(115:143, :) = 0;
    imshow(brightSpikes);
    title('Bright spikes other than central spike', 'FontSize', fontSize);
    
    promptMessage = sprintf('Now we will use these bright spikes to filter (mask) the spectrum.');
    button = questdlg(promptMessage, titleBarCaption, 'Continue', 'Cancel', 'Continue');
    if strcmpi(button, 'Cancel')
        return;
    end
    % Filter/mask the spectrum.
    frequencyImage(brightSpikes) = 0;
    % Take log magnitude so we can see it better in the display.
    amplitudeImage2 = log(abs(frequencyImage));
    minValue = min(min(amplitudeImage2))
    maxValue = max(max(amplitudeImage2))
    subplot(2, 3, 5);
    imshow(amplitudeImage2, [minValue maxValue]);
    axis on;
    title('Spikes zeroed out', 'FontSize', fontSize);
    % zoom(10)
    
    filteredImage = ifft2(fftshift(frequencyImage));
    amplitudeImage3 = abs(filteredImage);
    minValue = min(min(amplitudeImage3))
    maxValue = max(max(amplitudeImage3))
    subplot(2, 3, 6);
    imshow(amplitudeImage3, [minValue maxValue]);
    title('Filtered Image', 'FontSize', fontSize);
    % set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Maximize figure.
    % % 
    
end
% Spek 7
% Buatlah program yang melakukan motion bluring pada citra (grayscale dan berwarna),
% lalu lakukan dekonvolusi pada citra tersebut dengan penapis Wiener. Program penapis
% Wiener anda buat sendiri (tidak boleh menggunakan fungsi Wiener di dalam Matlab).
% Ujicoba pada dua citra di bawah ini dan dua citra tambahan:
