
% Reading some image data
% First, navigate MATLAB to the folder containing this script and the image data

% Prompt the user to select an image file
[fileName, filePath] = uigetfile({'*.jpg;*.png;*.bmp', 'Image files (*.jpg, *.png, *.bmp)'}, 'Select an image file to process');

% Check if the user canceled the operation
if isequal(fileName, 0)
    disp('User canceled the operation.');
    return; % Exit the routine
end

% Read the selected image file
imdat = imread(fullfile(filePath, fileName));

% Display the selected image
imagesc(imdat);
title('Selected Image');


% Display the original image
figure(1)
imagesc(imdat)


% Extract the red channel from the image
subplot(1,3,1)
imdat_red = imdat(:,:,1);
% Display the red channel
imagesc(imdat_red)
title('Red channel')
colorbar


% Extract the green channel from the image
subplot(1,3,2)
imdat_green = imdat(:,:,2);
% Display the green channel
imagesc(imdat_green)
title('Green channel')
colorbar

% Extract the blue channel from the image
subplot(1,3,3)
imdat_blue = imdat(:,:,3);
% Display the blue channel
imagesc(imdat_blue)
title('Blue channel')
colorbar

% 
subplot(1,3,1)
colormap('hot')

% Logical images and masks
% Create a logical mask where red is brighter than green
mask_red_ish = imdat_green < imdat_red;

% Display the mask showing pixels where red is brighter than green
figure (2)
imagesc(mask_red_ish)
title('Pixels in which red is brighter than green')
set(gca, 'fontSize', 14)
colorbar
axis equal 


% Calculate the area of the bluish mask
area_of_bluish_mask = sum(sum(mask_red_ish(:) ));

% Properties of regions identified by a mask
% Compute centroid, area, bounding box, convex image, and image properties of each region
stats = regionprops('table', mask_red_ish, 'Centroid', 'Area','BoundingBox', 'ConvexImage', 'Image');

% Extract centroid, area, bounding box, convex image, and image properties
mask_centres = stats.Centroid;
mask_areas   = stats.Area;
mask_boundingbox = stats.BoundingBox;
mask_conveximage = stats.ConvexImage; % same size as bounding box...
mask_image = stats.Image; % same size as bounding box...


% Plot locations of regions:
% Overlay the identified regions on the original image
figure (3)
imagesc(mask_red_ish)
axis equal
hold on
scatter(mask_centres(:,1), mask_centres(:,2), 'rx');
colorbar
hold off

% Throw away small regions
% Remove regions with an area less than 20000
listBad = (mask_areas<20000) ; % Coin area range
mask_centres(listBad==1,:)     = [];
mask_areas(listBad==1,:)       = [];
mask_boundingbox(listBad==1,:) = [];
mask_conveximage(listBad==1,:) = []; 
mask_image(listBad==1,:)       = []; 

% Throw away small regions
% Remove regions with an area larger than 70000
listBad = (mask_areas>70000) ; % Coin area range
mask_centres(listBad==1,:)     = [];
mask_areas(listBad==1,:)       = [];
mask_boundingbox(listBad==1,:) = [];
mask_conveximage(listBad==1,:) = []; 
mask_image(listBad==1,:)       = []; 

% Plot the mask after filtering out small regions
figure(4)
imagesc(mask_red_ish)
axis equal
hold on
scatter(mask_centres(:,1), mask_centres(:,2), 'rx');
colorbar
hold off


% Calculate average green brightness value
selected_region_green_brightness = zeros(size(mask_areas));

for i = 1:size(mask_areas, 1)
    % Get the center coordinates and radius for the current region (assuming circular region)
    center_x = mask_centres(i, 1);
    center_y = mask_centres(i, 2);
    radius = sqrt(mask_areas(i) / pi); % Calculate radius

    % Generate a grid of the same size as the image, calculate the distance of each pixel to the center
    [X, Y] = meshgrid(1:size(imdat_green, 2), 1:size(imdat_green, 1));
    distances_to_center = sqrt((X - center_x).^2 + (Y - center_y).^2);

    % Create a circular mask, set pixels within the radius near the center to 1, others to 0
    circle_mask = distances_to_center <= radius;

    % Extract circular region from the green channel image
    circle_region = imdat_green(circle_mask);

    % Calculate the average green brightness value for the circular region and divide by 256 to standardize
    selected_region_green_brightness(i) = mean(circle_region) / 256;
end



%% Bayes classifiers
% Problem: given the observed area and green reflected brighntess value
% Identify and classify Uk current coins in the images
a_range = 20000:100:70000;   % Possible observed coin areas
g_range = 0:0.002:1; % Possible observed green brightness value

[aa,gg] = meshgrid(a_range, g_range); % Possible observed area, green brightness

% Obtain prior information about coins from some training data that has 
% already been classified by someone:
% coin type
coin_types = {'2 pounds', '50 pence', '2 pence', '10 pence', '1 pound', '20 pence', '1 penny', '5 pence'};
% Population mean area of each coin
mean_area = [65371, 58331, 56243, 47367, 44482, 38033, 34150, 25526];
% Pop Variance (ss = 'sigma squared' of area of each coin
Var_area = [1361^2, 185^2, 1335^2, 249^2, 591^2, 835^2, 437^2, 567^2];
% Define mean green brightness values
mean_green_brightness = [0.623, 0.662, 0.440, 0.632, 0.599, 0.661, 0.467, 0.647];
% Define standard deviation values
Var_green_brightness= [0.053^2, 0.009^2, 0.041^2, 0.030^2, 0.033^2, 0.012^2, 0.038^2, 0.028^2];

% Invent a likelihood model for the observation:
% Probability of data (2 pounds):
pD_2pounds = exp(-(aa-mean_area(1)).^2/(2*Var_area(1))) .* exp(-(gg-mean_green_brightness(1)).^2/(2*Var_green_brightness(1))) ;
pD_2pounds = pD_2pounds./sum(pD_2pounds(:)); % Normalise by dividing by sum, since the equation for P(D|H_i) was not normalised.
pH_2pounds = 1/8; % Prior probability of 2pounds

% Probability of data (50 pence):
pD_50pence = exp(-(aa-mean_area(2)).^2/(2*Var_area(2))) .* exp(-(gg-mean_green_brightness(2)).^2/(2*Var_green_brightness(2))) ;
pD_50pence = pD_50pence./sum(pD_50pence(:));
pH_50pence = 1/8;

% Probability of data (2 pence):
pD_2pence = exp(-(aa-mean_area(3)).^2/(2*Var_area(3))) .* exp(-(gg-mean_green_brightness(3)).^2/(2*Var_green_brightness(3))) ;
pD_2pence = pD_2pence./sum(pD_2pence(:));
pH_2pence = 1/8;

% Probability of data (10 pence):
pD_10pence = exp(-(aa-mean_area(4)).^2/(2*Var_area(4))) .* exp(-(gg-mean_green_brightness(4)).^2/(2*Var_green_brightness(4))) ;
pD_10pence = pD_10pence./sum(pD_10pence(:));
pH_10pence = 1/8;

% Probability of data (1 pounds):
pD_1pounds = exp(-(aa-mean_area(5)).^2/(2*Var_area(5))) .* exp(-(gg-mean_green_brightness(5)).^2/(2*Var_green_brightness(5))) ;
pD_1pounds = pD_1pounds./sum(pD_1pounds(:));
pH_1pounds = 1/8;

% Probability of data (20 pence):
pD_20pence = exp(-(aa-mean_area(6)).^2/(2*Var_area(6))) .* exp(-(gg-mean_green_brightness(6)).^2/(2*Var_green_brightness(6))) ;
pD_20pence = pD_20pence./sum(pD_20pence(:));
pH_20pence = 1/8;

% Probability of data (1 pence):
pD_1penny = exp(-(aa-mean_area(7)).^2/(2*Var_area(7))) .* exp(-(gg-mean_green_brightness(7)).^2/(2*Var_green_brightness(7))) ;
pD_1penny = pD_1penny./sum(pD_1penny(:));
pH_1penny = 1/8;

% Probability of data (5 pence):
pD_5pence = exp(-(aa-mean_area(8)).^2/(2*Var_area(8))) .* exp(-(gg-mean_green_brightness(8)).^2/(2*Var_green_brightness(8))) ;
pD_5pence = pD_5pence./sum(pD_5pence(:));
pH_5pence = 1/8;

% Bayes rule, using total probabililty to get p(D)
p_2pounds_given_D = pD_2pounds .* pH_2pounds ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);
p_50pence_given_D = pD_50pence .* pH_50pence ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);
p_2pence_given_D = pD_2pence .* pH_2pence ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);
p_10pence_given_D = pD_10pence .* pH_10pence ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);
p_1pounds_given_D = pD_1pounds .* pH_1pounds ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);
p_20pence_given_D = pD_20pence .* pH_20pence ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);
p_1penny_given_D = pD_1penny .* pH_1penny ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);
p_5pence_given_D = pD_5pence .* pH_5pence ./ (pD_2pounds .* pH_2pounds + pD_50pence .* pH_50pence + pD_2pence .* pH_2pence + pD_10pence .* pH_10pence + pD_1pounds .* pH_1pounds + pD_20pence .* pH_20pence + pD_1penny .* pH_1penny + pD_5pence .* pH_5pence + 0.0001);


% Plot likelihood of observation given each coin type
figure(5)
mesh(aa, gg, p_2pounds_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of data given 2 pounds')

figure(6)
mesh(aa, gg, p_50pence_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of data given 50 pence')

figure(7)
mesh(aa, gg, p_20pence_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of data given 2 pence')


figure(8)
mesh(aa, gg, p_10pence_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of 10 pence given data')

figure(9)
mesh(aa, gg, p_1pounds_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of 1 pounds given data')

figure(10)
mesh(aa, gg, p_20pence_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of 20 pence given data')

figure(11)
mesh(aa, gg, p_1penny_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of 1 pence given data')

figure(12)
mesh(aa, gg, p_5pence_given_D);
xlabel('area')
ylabel('green value')
set(gca, 'fontSize', 14)
title('probability of 5 pence given data')

% Define the values of each coin type along with their units
coin_values = [2, 0.5, 0.02, 0.1, 1, 0.2, 0.01, 0.05]; % Values in pounds

% Assume coin_areas and green_brightness are extracted from each coin region
% Extract Coin Features
coin_areas = mask_areas;
coin_brightness = selected_region_green_brightness;

% Initialize the final coin types and probabilities
final_coin_types = cell(1, numel(coin_areas));
final_coin_probabilities = zeros(size(coin_areas));
final_coin_types1 = cell(1, numel(coin_areas));
final_coin_probabilities1 = zeros(size(coin_areas));
% Initialize variables for counting and totaling coin values
identified_coins = 0;
total_value = 0;

% Initialize counters for each coin type
coin_counts = zeros(1, length(coin_values));

% Iterate over each detected coin region
for i = 1:numel(coin_areas)
    % Map the area value to an index in the probability matrix
    [~, area_index] = min(abs(a_range - coin_areas(i)));
    % Map the green brightness value to an index in the probability matrix
    [~, brightness_index] = min(abs(g_range - coin_brightness(i)));

    % Get the probability of each coin type
    coin_probs = [p_2pounds_given_D(brightness_index, area_index), p_50pence_given_D(brightness_index, area_index), ...
                  p_2pence_given_D(brightness_index, area_index), p_10pence_given_D(brightness_index, area_index), ...
                  p_1pounds_given_D(brightness_index, area_index), p_20pence_given_D(brightness_index, area_index), ...
                  p_1penny_given_D(brightness_index, area_index), p_5pence_given_D(brightness_index, area_index)];

    % Normalize coin type probabilities
    coin_probs1 = coin_probs / sum(coin_probs);

    % Determine the coin type with the maximum probability
    [max_prob1, max_index1] = max(coin_probs1);
    [max_prob, max_index] = max(coin_probs);

    % Store the final coin type and probability
    final_coin_types{i} = coin_types{max_index};
    final_coin_probabilities(i) = max_prob;
    final_coin_types1{i} = coin_types{max_index1};
    final_coin_probabilities1(i) = max_prob1;

    % Increment the count of identified coins
    identified_coins = identified_coins + 1;

    % Determine the value of the identified coin
    coin_value = 0;
    if max_prob > 0
        coin_value = coin_values(max_index);
        % Increment the count for the identified coin type
        coin_counts(max_index) = coin_counts(max_index) + 1;
    end

    % Add the value of the identified coin to the total value
    total_value = total_value + coin_value;

    % Display the result for the current coin
    fprintf('Coin %d: Type: %s, Probability: %.2f, Value: %.2f pounds\n', i, final_coin_types{i}, final_coin_probabilities(i), coin_value);
end

% Display the total number of identified coins
fprintf('Number of Identified Coins:\n');
% Display coin types and their corresponding counts
for j = 1:length(coin_types)
    fprintf('The number of %s: %d\n', coin_types{j}, coin_counts(j));
end

% Display the total number of identified coins
fprintf('Total number of Identified Coins: %d \n', i);

% Display the total value of the identified coins
fprintf('Total Value of Identified Coins: %.2f pounds\n', total_value);


% Plotting Identified Regions:
% Display the original image
figure(13)
imagesc(imdat);
colormap(gray);
axis equal;
hold on;

% Plotting the identified regions and labeling them with coin values and probabilities
for i = 1:numel(coin_areas)
    % Define the coin value text and coin probabilities
    coin_value_text = sprintf('%s', final_coin_types{i});
    coin_value_probability = sprintf('(%.2f)', final_coin_probabilities(i));
    
    % Place the coin value text slightly above the centroid
    text(mask_centres(i, 1), mask_centres(i, 2), coin_value_text, 'Color', 'blue', 'FontSize',10, 'HorizontalAlignment', 'center');
    
    % Place the coin probability text below the coin value text
    text(mask_centres(i, 1), mask_centres(i, 2) + 50, coin_value_probability, 'Color', 'blue', 'FontSize',10, 'HorizontalAlignment', 'center');
end

hold off;

% Plotting Identified Regions:
% Display the original image
figure(14)
imagesc(imdat);
colormap(gray);
axis equal;
hold on;

% Plotting the identified regions and labeling them with coin values and normalized probabilities
for i = 1:numel(coin_areas)
    % Define the coin value text and coin probabilities
    coin_value_text = sprintf('%s', final_coin_types1{i});
    coin_value_probability = sprintf('(%.2f)', final_coin_probabilities1(i));
    
    % Place the coin value text slightly above the centroid
    text(mask_centres(i, 1), mask_centres(i, 2), coin_value_text, 'Color', 'blue', 'FontSize',10, 'HorizontalAlignment', 'center');
    
    % Place the coin probability text below the coin value text
    text(mask_centres(i, 1), mask_centres(i, 2) + 50, coin_value_probability, 'Color', 'blue', 'FontSize',10, 'HorizontalAlignment', 'center');
end

hold off;

% Initialize Total_probability
Total_probability = 1;
%the Total probability of certainty:
for i = 1:numel(coin_areas)

    Total_probability = Total_probability*final_coin_probabilities1(i);
end

% Display the Total probability of certainty:
fprintf('Total probability of certainty: %.2f \n', Total_probability);

