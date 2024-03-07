
#include <opencv2/opencv.hpp>

#include <iostream>
#include <limits>
#include <random>


// Style parameters
// I know, hardcoding parameters is illegal, but at some point I will hopefully create a GUI for this.
// The following parameters represent the "Impressionist" style of the original paper.
float threshold = 100;
std::vector<float> brushSizes = {8.0, 4.0, 2.0};
float curvatureFilter = 1.0;
float blurFactor = 0.5;
float opacity = 1.0;
float gridSize = 1.0;
int minStrokeLength = 4;
int maxStrokeLength = 16;
float colorValueJitter = 0.0;
float colorPaletteJitter = 0.0;



typedef struct {
    int brushRadius;
    std::vector<cv::Point_<float>> controlPoints;
    cv::Vec<float, 3> color;
} Stroke;


cv::Mat rawImage;
cv::Mat originalImage;
cv::Mat blurredImage;
cv::Mat diffImage;
cv::Mat luminance;
cv::Mat Dx;
cv::Mat Dy;
cv::Mat canvas;

// Additional images to implement opacity setting using cv::addWeighted function
cv::Mat tempCanvas;
cv::Mat overlay;

// The original paper states:
// "In order to cover the canvas with paint, the
// canvas is initially painted a special “color” C such that the
// difference between C and any color is MAXINT."
// I have no idea what this magic color should be when we work
// with a uint8 RGB image, so I keep track of the pixels that have already
// been painted separately.
cv::Mat alreadyPaintedMask;



uint32_t colorDiff(const cv::Vec3b &lhs, const cv::Vec3b &rhs) {
    return sqrt((lhs[0] - rhs[0]) * (lhs[0] - rhs[0]) + (lhs[1] - rhs[1]) * (lhs[1] - rhs[1]) + (lhs[2] - rhs[2]) * (lhs[2] - rhs[2]));
}

double getRandomNumber() {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(-1.0 , 1.0);
    return distribution(generator);
}



Stroke makeSplineStroke(const cv::Point_<double> &startPoint, float brushSize) {
    Stroke resultStroke;

    auto strokeColor = blurredImage.at<cv::Vec3b>(startPoint);

    // Apply Jitter
    // Color value jitter generates one value that is applied to the r, g ang b channels equally
    // Color palette jitter generates a seperate value for r, g and b each
    // Add the sum to each channel and make sure to stay in the uint8 range

    float jv = colorValueJitter * getRandomNumber();
    float jb = colorPaletteJitter * getRandomNumber() + jv;
    float jg = colorPaletteJitter * getRandomNumber() + jv;
    float jr = colorPaletteJitter * getRandomNumber() + jv;

    int newB = int(jb * 255.0) + strokeColor[0];
    int newG = int(jg * 255.0) + strokeColor[1];
    int newR = int(jr * 255.0) + strokeColor[2];

    strokeColor[0] = uint8_t(std::clamp(newB, 0, 255));
    strokeColor[1] = uint8_t(std::clamp(newG, 0, 255));
    strokeColor[2] = uint8_t(std::clamp(newR, 0, 255));

    resultStroke.color = strokeColor;
    resultStroke.brushRadius = brushSize;
    resultStroke.controlPoints = {startPoint};

    cv::Point_<double> currentPoint = startPoint;
    cv::Point_<double> lastDirection = {0.0, 0.0};

    for(int i = 1; i <= maxStrokeLength; i++) {
        cv::Point currentAccessPoint = {int(currentPoint.x), int(currentPoint.y)};

        // So the difference between blurredImageColor and canvasColor should be int max when
        // the pixel at currentAccessPoint has not been painted yet
        auto blurredImageColor = blurredImage.at<cv::Vec3b>(currentAccessPoint);
        auto canvasColor = canvas.at<cv::Vec3b>(currentAccessPoint);

        uint32_t canvasDiff = std::numeric_limits<uint32_t>::max();
        if(alreadyPaintedMask.at<uint8_t>(currentAccessPoint) != 0) {
            canvasDiff = colorDiff(blurredImageColor, canvasColor);
        }

        if(i > minStrokeLength && (canvasDiff < colorDiff(blurredImageColor, strokeColor))) {
            return resultStroke;
        }

        double dx = Dx.at<double>(currentAccessPoint);
        double dy = Dy.at<double>(currentAccessPoint);

        // Detect vanishing gradient
        if((dx * dx + dy * dy) == 0) {
            dx = getRandomNumber();
            dy = getRandomNumber();
        }

        // Get normal vector (orthogonal to gradient)
        cv::Point_<double> currentDirection = {-dy, dx};

        // If necessary, reverse direction
        if(lastDirection.dot(currentDirection) < 0) {
            currentDirection = -currentDirection;
        }

        currentDirection /= sqrt(currentDirection.x * currentDirection.x + currentDirection.y * currentDirection.y); // unit length

        // Filter direction
        currentDirection = curvatureFilter * currentDirection + (1.0 - curvatureFilter) * lastDirection;

        // Finalize position of new control point
        currentDirection /= sqrt(currentDirection.x * currentDirection.x + currentDirection.y * currentDirection.y);
        currentPoint = {currentPoint.x + brushSize * currentDirection.x, currentPoint.y + brushSize * currentDirection.y};
        lastDirection = currentDirection;

        // Prevent going outside of the image
        if(currentPoint.x < 0 || currentPoint.x >= canvas.cols || currentPoint.y < 0 || currentPoint.y >= canvas.rows) {
            return resultStroke;
        }

        resultStroke.controlPoints.push_back(currentPoint);
    }

    return resultStroke;
}



float coxDeBoor(float t, int i, int k, const std::vector<int>& tVector) {
    if(k == 1) {
        if(tVector[i] <= t && t < tVector[i + 1]) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else {
        float part1, part2;
        if(coxDeBoor(t, i, k - 1, tVector) != 0) {
            part1 = ((t - tVector[i]) / (tVector[i + k - 1] - tVector[i])) * coxDeBoor(t, i, k - 1, tVector);
        }
        else {
            part1 = 0.0;
        }
        if(coxDeBoor(t, i + 1, k - 1, tVector) != 0) {
            part2 = (tVector[i + k] - t) / (tVector[i + k] - tVector[i + 1]) * coxDeBoor(t, i + 1, k - 1, tVector);
        }
        else {
            part2 = 0.0;
        }
        return part1 + part2;
    }
}


cv::Point_<float> bsplineCurve(float t, int k, const std::vector<int>& tVector, const std::vector<cv::Point_<float>>& points) {
    cv::Point_<float> result = {0.0, 0.0};
    for(int i = 0; i < points.size(); i++) {
        result += coxDeBoor(t, i, k, tVector) * points[i];
    }
    return result;
}


void drawStroke(Stroke& stroke) {
    // Fancy B-Spline stuff
    int k = 4;
    // We need to have at least k control points, otherwise this bspline function generates only the point (0,0)
    // Therefore repeat last point until we have enough
    while(stroke.controlPoints.size() < k) {
        stroke.controlPoints.push_back(stroke.controlPoints.back());
    }
    int n = stroke.controlPoints.size();
    std::vector<int> tVector = {};
    for(int i = 0; i < k; i++) {
        tVector.push_back(0);
    }
    for(int i = 0; i < (n + k) - k - k; i++) {
        tVector.push_back(i + 1);
    }
    for(int i = 0; i < k; i++) {
        tVector.push_back(n - k + 1);
    }

    float max = float(tVector[n]);
    for(float t = float(tVector[k - 1]); t <= max; t += 0.1) {
        cv::Point_<float> currentPoint = bsplineCurve(t, k, tVector, stroke.controlPoints);
        cv::circle( overlay,
            {int(currentPoint.x), int(currentPoint.y)},
            stroke.brushRadius,
            stroke.color,
            cv::FILLED,
            cv::LINE_AA ); // Anti-Aliasing is only performed when target image is type CV_8U...
        cv::circle( alreadyPaintedMask,
            {int(currentPoint.x), int(currentPoint.y)},
            stroke.brushRadius,
            255,
            cv::FILLED,
            cv::LINE_AA ); // Anti-Aliasing is only performed when target image is type CV_8U...
        
    }
}


void paintLayer(float brushSize) {

    tempCanvas = canvas.clone();
    overlay = canvas.clone();

    std::vector<Stroke> allStrokes = {};

    cv::absdiff(canvas, blurredImage, diffImage);
    cv::cvtColor(diffImage, diffImage, cv::COLOR_BGR2GRAY);

    int stepsize = int(gridSize * brushSize);
    for(int x = 0; x < canvas.cols; x += stepsize) {
        for(int y = 0; y < canvas.rows; y += stepsize) {

            uint32_t areaError = 0;
            cv::Point largestErrorPoint;
            uint32_t maxDiff = std::numeric_limits<uint32_t>::lowest();

            for(int mx = x - stepsize / 2; mx <= x + stepsize / 2; mx++) {
                for(int my = y - stepsize / 2; my <= y + stepsize / 2; my++) {
                    int clampedmx = std::clamp(mx, 0, canvas.cols - 1);
                    int clampedmy = std::clamp(my, 0, canvas.rows - 1);

                    uint32_t currentDiff = std::numeric_limits<uint16_t>::max(); // So that we can add a bunch of int max values together, we take the max value of uint16 and contain it in a uint32
                    // Now if the current pixel is already painted on the target image, we take the acutal difference, otherwise
                    // leave the difference at int max.
                    if(alreadyPaintedMask.at<uint8_t>(clampedmy, clampedmx) != 0) {
                        currentDiff = diffImage.at<uint8_t>(clampedmy, clampedmx);
                    }
                    
                    if(currentDiff > maxDiff) {
                        maxDiff = currentDiff;
                        largestErrorPoint = {clampedmx, clampedmy};
                    }
                    areaError += currentDiff;
                }
            }
            areaError /= stepsize * stepsize;

            if(areaError > threshold) {
                allStrokes.push_back(makeSplineStroke(largestErrorPoint, brushSize));
            }
        }
    }

    std::random_shuffle(allStrokes.begin(), allStrokes.end());

    int strokeCounter = 0;

    for(auto &currentStroke : allStrokes) {
        drawStroke(currentStroke);
        strokeCounter++;
        if(strokeCounter >= 100) {
            cv::addWeighted(overlay, opacity, tempCanvas, 1.0 - opacity, 0.0, canvas);
            cv::imshow("Painterly Image", canvas);
            cv::waitKey(1);
            strokeCounter = 0;
        }
    }
}



int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        std::cout << "usage: PainterlyRendering.exe <Image_Path>\n";
        return EXIT_FAILURE;
    }

    rawImage = cv::imread( argv[1], 1 );
    if ( !rawImage.data )
    {
        std::cout << "No image data \n";
        return -1;
    }

    rawImage.convertTo(originalImage, CV_8U);

    canvas = cv::Mat::zeros(originalImage.rows, originalImage.cols, CV_8UC3);
    canvas.forEach<cv::Vec3b>([&](cv::Vec3b &value, const int position[]) {
        value = {255, 255, 255};
    });
    
    alreadyPaintedMask = cv::Mat::zeros(originalImage.rows, originalImage.cols, CV_8UC1);

    cv::namedWindow("Painterly Image", cv::WINDOW_AUTOSIZE );

    for(auto& currentBrushSize : brushSizes) {
        cv::GaussianBlur(originalImage, blurredImage, cv::Size(15, 15), blurFactor * currentBrushSize);
        cv::cvtColor(blurredImage, luminance, cv::COLOR_BGR2GRAY);
        cv::Sobel(luminance, Dx, CV_64F, 1, 0, 5);
        cv::Sobel(luminance, Dy, CV_64F, 0, 1, 5);
        paintLayer(currentBrushSize);
    }

    cv::imwrite("image.png", canvas);

    return 0;
}