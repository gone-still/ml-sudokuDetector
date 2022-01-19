% File        :   main.oy (fastKLT-tracker Python Version)
% Version     :   1.0.0
% Description :   Implements Zana Zakaryaie's FAST-KLT tracker originally written in C++
% Date:       :   Apr 03, 2015
% Author      :   Ricardo Acevedo-Avila
% License     :   CC0 1.0 Universal

% Clear all workpsace
close all;
clc;
clear all;

% Path to resources:
path = '.\resources\';

% Debug Variables (1/0)
showFigures = 1;
showTrainData = 1;

% Script Variables:
symbolArea = 0;
biggestArea = 0;
blobMinX = 0;
blobMinY = 0;
blobWidth = 0;
blobHeight = 0;
boardLabel = 0;

% Number of Sudoku cells:
matrixCells = 9; 

% Train image setup,
% The training dataset is unique image of
% trainRows x trainCols Cells:
trainRows = 9;
trainCols = 5;

% Train matrix data:
features = 3;
samples = 9;

% Misc tuning:
threshSensitivity = 0.05; %"Sensitivity" of thresholding

% Load images:
puzzleRGB = imresize(imread([path 'puzzleSampleRGB04.png']), 1);
trainData = imread([path 'numbersTrainData02.png']);

figure, imshow(puzzleRGB);

% RGB 2 Gray conversion:
puzzleGray = rgb2gray(puzzleRGB);

[imageHeight imageWidth] = size(puzzleGray);

% Threshold level automatic computation:
thresholdLevel = graythresh(puzzleGray);
thresholdLevel = thresholdLevel - threshSensitivity * graythresh(puzzleGray);

puzzleBin = im2bw(puzzleGray,thresholdLevel);


% Fills holes in the (inverted) binary image:
puzzleFilled = imfill(~puzzleBin, 'holes');

if (showFigures) 
	figure, imhist(puzzleGray);
	figure, imshow(puzzleBin);
    figure, imshow(~puzzleBin);
	figure, imshow(puzzleFilled)
	hold on;
end;

% Board detection:
% Search for the biggest blob:
[labels, objects] = bwlabel(puzzleFilled);

for blobIndex = 1:objects

	[r c] = find(labels == blobIndex);

	% Blob centroid:
	x = mean(c);
	y = mean(r);

	blobRows = size(r, 1);
	blobCols = size(c, 1);
	symbolArea = blobRows * blobCols;
    
	if (symbolArea > biggestArea)

		if (showFigures)

			% Centroid plot:
			plot(x, y, 'Marker', 'o', 'MarkerEdgeColor', 'r', ...
				'MarkerFaceColor', 'k', 'MarkerSize', 10);
			plot(x, y, 'Marker', '*', 'MarkerEdgeColor', 'g');

		end;

		% Board blob info:
		biggestArea = symbolArea; 
		boardLabel = blobIndex;

	end 
    
end

% Isolate board given its blob index:
puzzleBoard = zeros(imageHeight, imageWidth);

% Iterate thru label matrix:
for maskRow = 1:imageHeight

	for maskCol = 1:imageWidth

		% Extract board based on index:
		label = labels( maskRow, maskCol );
		if (label == boardLabel)
			% Create board binary mask:
			puzzleBoard( maskRow, maskCol ) = 1;
		end

	end

end

% Board expansion:
SE = strel('square',3);
boardDilate = imdilate(puzzleBoard,SE);

% Board Edges:
boardEdges = imdilate(imdilate(edge(boardDilate,'sobel'), SE), SE); 

% Board line detection via Hough Transform:
[ H theta rho ] = hough(boardEdges);
maxPeaks = 10;
peakThreshold = ceil( 0.5 * max( H(:) ) );

P = houghpeaks(H, maxPeaks, 'threshold', peakThreshold);

x = theta( P(:,2) );
y = rho( P(:,1) );

% Apply the line detector:
lines = houghlines(boardEdges, theta, rho, P, 'FillGap', 100, 'MinLength', 200);

% Get detected lines:
totalLines = length(lines);
points = zeros(2*totalLines, 2);

if (showFigures)
	figure, imshow(puzzleBoard);
	figure, imshow(boardDilate);
	figure, imshow(boardEdges);
	hold on;
end

% Plot the detected lines:
for lineIndex = 1:totalLines
    xy = [ lines(lineIndex).point1; lines(lineIndex).point2 ];
    rowEnd = 2*lineIndex;
    rowStart = rowEnd - 1;
    points(rowStart:rowEnd,1:2) = xy;
    if (showFigures)
        plot( xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green' );
        plot( xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow' );
        plot( xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red' );
    end;

end;

% Corner detection via k-means (clusters the line points into one point):
totalClusters = 4;
[idx clusterCentroid] = kmeans(points,totalClusters,'Distance','sqeuclidean', 'Display','final', 'Replicates',5);

% Plot big red marks on centroid location:
if (showFigures)
	plot(clusterCentroid(:,1),clusterCentroid(:,2),'rx', 'MarkerSize',15,'LineWidth',3)
end;

% Region division:
imageHalfWidth = 0.5 * imageWidth;
imageHalfHeight = 0.5 * imageHeight;
pointsIn = zeros(totalClusters,2);

for clusterIndex = 1:totalClusters

	cx = clusterCentroid( clusterIndex, 1 );
	cy = clusterCentroid( clusterIndex, 2 );

	if (cy < imageHalfHeight)

		% Quad 1
		if (cx < imageHalfWidth)
			quadIndex = 1;
		% Quad 2
		else
			quadIndex = 2;            
		end

	else

		% Quad 3
		if (cx < imageHalfWidth)
			quadIndex = 3;  
		% Quad 4
		else
			quadIndex = 4; 
		end
	end

	% Points sorting:
	pointsIn(quadIndex,1) = cx;
	pointsIn(quadIndex,2) = cy;  
end

% Perspective correction:

% Square approximation:
xMin = min(pointsIn(:,1));
yMin = min(pointsIn(:,2));

xMax = max(pointsIn(:,1));
yMax = max(pointsIn(:,2));

rectWidth = xMax - xMin;
rectHeight = yMax - yMin;   

rect = [pointsIn(1,1) pointsIn(1,2); 
        pointsIn(1,1) + rectWidth pointsIn(1,2); 
        pointsIn(1,1) pointsIn(1,2)+ rectHeight; 
        pointsIn(1,1) + rectWidth pointsIn(1,2) + rectHeight];

pointsOut = rect; 

% Apply transformation:
tform = cp2tform(pointsIn,pointsOut, 'projective');
correctedImage = imtransform(puzzleGray,tform);

% Show rectified zone:
if (showFigures)
	rect = [pointsIn(1,1) pointsIn(1,2) rectWidth rectHeight];
	rectangle('Position',rect,'EdgeColor','r');
	hold off; 
end;

% Crop rectified zone:
[rectImageHeight rectImageWidth] = size(correctedImage);

wRatio = abs(1 - rectImageWidth/imageWidth);
hRatio = abs(1 - rectImageHeight/imageHeight);

adjX = pointsIn(1,1) + wRatio * pointsIn(1,1) + 2;
adjY = pointsIn(1,2) + hRatio * pointsIn(1,2) + 7;

adjW = rectWidth + 2;
adjH = rectHeight + 5;

rect2 = [adjX adjY adjW adjH] ; 
puzzleCropped = imcrop(correctedImage, rect2); 

binImage = ~im2bw(puzzleCropped, thresholdLevel);
%binImage = imfilter(binImage, fspecial( 'unsharp' ));

% Number isolation: binImage - bigBoard
bigArea = bwareaopen(binImage, 1000);
puzzleNumbers = binImage - bigArea;
puzzleNumbers = bwareaopen(puzzleNumbers, 70);

% Prepare the attibute container for traning:
[rows cols] = size(trainData);

cellHeight = floor(rows/trainRows);
cellWidth = floor(cols/trainCols);

trainMeasures = zeros(samples, features);

if (showFigures)
    figure, imshow(correctedImage);
    rectangle('Position',rect2,'EdgeColor','r');
    figure, imshow(puzzleCropped);
    figure, imshow(binImage);
    figure, imshow(puzzleNumbers);
    imwrite( puzzleNumbers, [path 'testImage.png'] );
end;

if (showTrainData)
    figure, imshow(trainData);
	figure;
end;

%======================================
% Training ============================
%======================================

for i = 1:trainRows

	for j = 1:trainCols

		% Cell coordinates:
		cellX = (j-1)*cellWidth + 1;
		cellY = (i-1)*cellHeight + 1;

		currentWidth = cellWidth - 1;
		currentHeight = cellHeight - 1;

		% Current cell coordinates:
		rect = [cellX cellY currentWidth currentHeight] ;         

		% Current subimage:
		currentCell = imcrop(trainData, rect);
		% Threshold image:
		currentCell = im2bw(currentCell, 130/255);
		
		% Number Skelleton:
		cellSkell = bwmorph(currentCell,'skel',5);       
		if (showTrainData)
		    imshow(cellSkell);
		end;

		% Extract attibutes from current cell:        
		[labels objects] = bwlabel(cellSkell);
		blobProperties = regionprops(labels, 'EulerNumber');

		if (objects > 1)
		    disp('More objects than expected!')
		    break;
		end

		% Compute some basic attributes:
		% Hu Moments:

		I = cellSkell;

		x = 1:cellWidth;
		y = 1:cellHeight;

		[X, Y] = meshgrid(x,y);

		M00 = sum(sum(I));
		M10 = sum(sum(X.*I));
		M01 = sum(sum(Y.*I));

		xAverage = M10./M00;
		yAverage = M01./M00;

		X = X - xAverage;
		Y = Y - yAverage;      

		if (showTrainData)
		    text(xAverage,yAverage,num2str(i*j),'Color',[1 0 0]);   
		end;

		% Statistical Moments:
		M11 = sum(sum(X.*Y.*I));
		M20 = sum(sum(X.^2.*I));
		M02 = sum(sum(Y.^2.*I));

		M21 = sum(sum(X.^2.*Y.*I));
		M12 = sum(sum(X.*Y.^2.*I));

		M30 = sum(sum(X.^3.*I));
		M03 = sum(sum(Y.^3.*I));

		eta11 = M11./M00.^2;
		eta20 = M20./M00.^2;
		eta02 = M02./M00.^2;

		eta21 = M21./M00.^(5./2);
		eta12 = M12./M00.^(5./2);
		eta30 = M30./M00.^(5./2);

		eta03 = M02./M00.^(5./2);

		hu01 = eta20 + eta02;
		hu02 = (eta20 - eta02).^2 + (2.*eta11).^2;
		hu03 = (eta30 - 3.*eta12).^2 + (3.*eta21 - eta03).^2;

		skewnessX = M30./M20.^(3./2);
		skewnessY = M03./M02.^(3./2);

		hu1Vector(i, j) = hu01;
		%hu2Vector(i, j) = hu02;
		%hu3Vector(i, j) = hu03;

		blobEuler(i,j) = blobProperties.EulerNumber;

		skX(i,j) = skewnessX;

	end    
end

% Prototype construction:
trainMeasures(:,1) = mean(hu1Vector, 2) ./ max(mean(hu1Vector, 2));
trainMeasures(:,2) = mean(blobEuler, 2);
trainMeasures(:,3) = mean(skX, 2);

classLabels = 1:9;

featureList = trainMeasures;

% Plot some nice figures with all the computed attributes:
if (showFigures)

	% Hu01 vs Euler:
    figure;
    plot(featureList(:,1), featureList(:,2), 'ro');
    hold on;
    xlabel('Hu01');
    ylabel('Euler');

	plotLabels = num2str((classLabels)','%d');

	text(featureList(:,1), featureList(:,2), plotLabels, 'horizontal','left', 'vertical','bottom');
    hold off;

	% Hu01 vs Skewness X:
    figure;
    plot(featureList(:,1), featureList(:,3), 'ro');
    hold on;
    xlabel('Hu01');
    ylabel('skX');

    text(featureList(:,1), featureList(:,3), plotLabels, 'horizontal','left', 'vertical','bottom');
    hold off;

end;

%======================================
% Test  ===============================
%======================================

numbers = imresize(puzzleNumbers, 1);
numbersSkel = bwmorph(numbers,'skel',5);

% Blob detection:
[labels objects] = bwlabel(numbersSkel);
[imageRows imageCols] = size(numbersSkel);

% Attributes:
blobProperties = regionprops(labels, 'Area', 'Perimeter', 'EulerNumber');

% Numeric Labels:
numLabels = zeros(1, objects);

% Label & centroid storage:
% 3 columns: label, cx, cy
dataCentroids = zeros(objects, 3);

if (showFigures)
    figure, imshow(numbers);
    figure, imshow(numbersSkel);
end;

for k = 1:objects 
    
	% Individual blob label storage:
	numLabels(1, k) = k;  

	% Current blob:
	I = zeros(imageRows, imageCols);
	[rows cols] = size(I);

	% Blob of interest:
	index = find(labels == k);
	I(index) = 1;

	x = 1:cols;
	y = 1:rows;

	[X, Y] = meshgrid(x,y);

	M00 = sum(sum(I));
	M10 = sum(sum(X.*I));
	M01 = sum(sum(Y.*I)); 

	xAverage = M10./M00;
	yAverage = M01./M00;

	X = X - xAverage;
	Y = Y - yAverage;      

	if (showFigures)
		text(xAverage,yAverage,num2str(k),'Color',[1 0 0]);
	end;

	dataCentroids(k, 1) = k;
	dataCentroids(k, 2) = xAverage;
	dataCentroids(k, 3) = yAverage;

	M11 = sum(sum(X.*Y.*I));
	M20 = sum(sum(X.^2.*I));
	M02 = sum(sum(Y.^2.*I));

	M21 = sum(sum(X.^2.*Y.*I));
	M12 = sum(sum(X.*Y.^2.*I));

	M30 = sum(sum(X.^3.*I));
	M03 = sum(sum(Y.^3.*I));

	eta11 = M11./M00.^2;
	eta20 = M20./M00.^2;
	eta02 = M02./M00.^2;

	eta21 = M21./M00.^(5./2);
	eta12 = M12./M00.^(5./2);
	eta30 = M30./M00.^(5./2);

	eta03 = M02./M00.^(5./2);

	hu01 = eta20 + eta02;
	hu02 = (eta20 - eta02).^2 + (2.*eta11).^2;
	hu03 = (eta30 - 3.*eta12).^2 + (3.*eta21 - eta03).^2;

	skewnessX = M30./M20.^(3./2);

	hu1VectorSample(k,1) = hu01;
	%hu2VectorSample(k,1) = hu02;
	%hu3VectorSample(k,1) = hu03;

	eulerSample(k, 1) = blobProperties(k).EulerNumber; 

	sampleSkewX(k, 1) = skewnessX;
    
end

% Data classification:
sampleFeatureList = zeros(objects, features);

sampleFeatureList(:,1) = hu1VectorSample ./ max(hu1VectorSample);
sampleFeatureList(:,2) = eulerSample;
sampleFeatureList(:,3) = sampleSkewX;

% Function call to the classifier:
classifierResults = knnclassify(sampleFeatureList, trainMeasures, classLabels, 1);

if (showFigures)
	figure,
	imshow(~numbers);

	hold on;
end;

% Numerical matrix position aproximation to get a 
% nice numerical matrix:
puzzleMatrix = zeros(matrixCells,matrixCells);
numberCellWidth = fix(imageCols/matrixCells);
numberCellHeight = fix(imageRows/matrixCells);

for k = 1:objects
    
    classifiedObject = classifierResults(k); 
    
    % Centroids:
    cx = dataCentroids(k,2);
    cy = dataCentroids(k,3);
    
    % Number plotting:    
    if (showFigures)
        text(cx,cy,num2str(classifiedObject),'Color',[1 0 0], 'FontSize', 15);
    end;

    % Insertion into numerical matrix:
    colIndex = ceil(cx/numberCellWidth);
    rowIndex = ceil(cy/numberCellHeight);
    
    puzzleMatrix(rowIndex, colIndex) = classifiedObject;   
    
end

% Show the final matrix:
puzzleMatrix

