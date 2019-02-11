//
//  ViewController.m
//  ios_mx_det
//
//  Created by Weixing Zhang on 02/05/2019.
//  Copyright © 2019 Weixing Zhang. All rights reserved.
//

#import "ViewController.h"

//
const mx_float DEFAULT_MEAN = 110;
const mx_float DEFAULT_STD = 58;
cv::Scalar LineColor = cvScalar(0,255,0); //overlay color

// Model filenames
std::string model_dir="/model/";
std::string json_filename = model_dir+"mobilenet0.5_224-symbol"; //.json
std::string param_filename = model_dir+"mobilenet0.5_224-0000"; //.param

// CHW by default
bool usingHWC = false;

// Image size and channels
int width = 224;
int height = 224;
int channels = 3;

// MXPred parameters
int dev_type = 1;  // 1: cpu, 2: gpu
int dev_id = 0;
mx_uint num_input_nodes = 1;  // 1 for feedforward
const char* input_key[1] = {"data"};
const char** input_keys = input_key;
const mx_uint input_shape_indptr[2] = { 0, 4 };
mx_uint input_shape_data[4]={1,1,1,1};

clock_t lastEnd=NULL, thisEnd;

// Channel conversion helper function
void CHW_to_HWC(mx_float* image_data, mx_float* new_image,
                const int channel, const int height, const int width)
{
    for (int i=0;i<height;++i)
    {
        for (int j=0;j<width;++j)
        {
            for (int k=0;k<channel;++k)
            {
                *new_image++=*(image_data+(channel-1-k)*height*width+j+i*width);
            }
        }
    }
}

// Softmax function
void softmax(const std::vector<mx_float> input_vector, mx_float* softmax_prob, const int size)
{
    mx_float den=0;
    for (int i=0;i<size;++i)
    {
        den+=std::exp(input_vector[i]);
    }
    for (int j=0;j<size;++j)
    {
        *softmax_prob++=std::exp(input_vector[j])/den;
    }
}

@interface ViewController()
{
    
    CvVideoCamera* videoCamera;
}

@property (nonatomic, retain) CvVideoCamera* videoCamera;

@end

@implementation ViewController
- (void)viewDidLoad
{
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
//    UIAlertView * alert = [[UIAlertView alloc] initWithTitle:@"Hello!" message:@"mxnet&opencv up and running!" delegate:self cancelButtonTitle:@"Continue" otherButtonTitles:nil];
//    [alert show];

    // Do any additional setup after loading the view, typically from a nib.
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:_ImageOutput];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
    self.videoCamera.grayscaleMode = NO;
    
    // Set data shape
    if (usingHWC)
    {
        input_shape_data[1]=static_cast<mx_uint>(height);
        input_shape_data[2]=static_cast<mx_uint>(width);
        input_shape_data[3]=static_cast<mx_uint>(channels);
    }
    else
    {
        input_shape_data[1]=static_cast<mx_uint>(channels);
        input_shape_data[2]=static_cast<mx_uint>(height);
        input_shape_data[3]=static_cast<mx_uint>(width);
    }
    
    clock_t startTime,endTime;
    
    // Load model
    startTime = clock();
    NSString *json_path = [[NSBundle mainBundle] pathForResource:@(json_filename.c_str()) ofType:@"json"];
    NSString *param_path = [[NSBundle mainBundle] pathForResource:@(param_filename.c_str()) ofType:@"params"];
    model_symbol = [[NSString alloc] initWithData:[[NSFileManager defaultManager] contentsAtPath:json_path] encoding:NSUTF8StringEncoding];
    model_params = [[NSFileManager defaultManager] contentsAtPath:param_path];
    assert(0==MXPredCreate([model_symbol UTF8String],
                           [model_params bytes],
                           (int)[model_params length],
                           dev_type,
                           dev_id,
                           num_input_nodes,
                           input_keys,
                           input_shape_indptr,
                           input_shape_data,
                           &pred_hnd));
    assert(pred_hnd);
    endTime = clock();

    double init_model_costs = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    std::cout << "init model costs : " << init_model_costs << "ms" << std::endl;


}

#pragma mark - Protocol CvVideoCameraDelegate

#ifdef __cplusplus
- (void)processImage:(cv::Mat&)image;
{
    // Do some amazing stuff with the frame here
    
    // Resizing to
    cv::Mat image_input;
    cvtColor(image, image, cv::COLOR_BGR2RGB);
    resize(image, image, cv::Size(480, 640));
    
    if (nnSwitch.isOn)
    {
        resize(image, image_input, cv::Size(width, height));
        
        // Construct image tensor
        int image_size = width * height * channels;
        std::vector<mx_float> image_data(image_size);
        
        // De-interleave and normalize
        // b g r ?
        unsigned char *ptr = image_input.ptr();
        float *data_ptr = image_data.data();
        float mean_b, mean_g, mean_r;
        float std_b, std_g, std_r;
        mean_b = mean_g = mean_r = DEFAULT_MEAN;
        std_b = std_g = std_r = DEFAULT_STD;
        
        for (int i = 0; i < image_size; i +=3) {
            *(data_ptr++) = (static_cast<float>(ptr[i]) - mean_r)/DEFAULT_STD;
        }
        for (int i = 1; i < image_size; i +=3) {
            *(data_ptr++) = (static_cast<float>(ptr[i]) - mean_g)/DEFAULT_STD;
        }
        for (int i = 2; i < image_size; i +=3) {
            *(data_ptr++) = (static_cast<float>(ptr[i]) - mean_b)/DEFAULT_STD;
        }
        
        // Swap axes if needed
        std::vector<mx_float> new_image(image_size);
        if (usingHWC)
        {
            CHW_to_HWC(image_data.data(), new_image.data(), channels, height, width);
        }
        else
        {
            new_image.clear();
        }
        
        // Bind tensor to model
        MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));
        
        // Forward net
        MXPredForward(pred_hnd);
        
        // Get output shape before retrieving output
        mx_uint output_index = 0; // class id
        mx_uint *shape = NULL;
        mx_uint shape_len = 0;
        MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);
        mx_uint size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
        
        // Get network output
        std::vector<mx_float> output_data_0(size);
            // MXNDArrayWaitToRead(&(output_data_0[0]));
        MXPredGetOutput(pred_hnd, output_index, output_data_0.data(), size);
        
        // Get predicted class
        std::vector<mx_float> softmax_output(size);
        softmax(output_data_0, softmax_output.data(), size);
        int max_index=0;
        for (int i=0;i<size;++i) if (output_data_0[i]>output_data_0[max_index]) max_index=i;
        
        // Overlay result
        cv::putText(image,
                    ClassMap[max_index],
                    cvPoint(20, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1, LineColor,
                    2);
        cv::putText(image,
                    std::to_string(softmax_output[max_index]),
                    cvPoint(20, 620),
                    cv::FONT_HERSHEY_SIMPLEX, 1, LineColor,
                    2);
    }
    
//    cv::Rect myROI(10, 10, 100, 100);
//    // Crop the full image to that image contained by the rectangle myROI
//    image = image(myROI);
    
    // Overlay FPS
    if (lastEnd)
    {
        thisEnd=clock();
        double frame_time= (double)(thisEnd - lastEnd) / CLOCKS_PER_SEC;
        std::string strFPS = std::to_string(1/frame_time);
        cv::putText(image,
                    "FPS:"+strFPS,
                    cvPoint(300, 620),
                    cv::FONT_HERSHEY_SIMPLEX, 1, LineColor,
                    2);
    }
    lastEnd=clock();
}
#endif

#pragma mark - UI Actions

- (IBAction)StartButton:(id)sender
{
    [sender setHidden:YES];
    [self.videoCamera start];
    [nnSwitch setHidden:NO];
    UIAlertView * alert = [[UIAlertView alloc] initWithTitle:@"Check that out" message:@"Camera up!" delegate:self cancelButtonTitle:@"Continue" otherButtonTitles:nil];
    [alert show];
}

- (IBAction)on_off:(id)sender {
}
@end

