// ------------------------- OpenPose Library Tutorial - Wrapper - Example 3 - Asynchronous Output -------------------------
//비동기 출력 모드: 성능이 문제가 되지 않고 사용자가 출력 OpenPose 형식을 사용하려는 경우 빠른 프로토타이핑에 이상적임 
//사용자들은 그가 원할 때 OpenPose 포장지로부터 가공된 프레임을 얻는다는 것을 암시한다


// 이 예는 사용자에게 OpenPose 래퍼 클래스를 사용하는 방법을 보여준다.
	// 1. 이미지/비디오/웹캠 폴더 읽기
	// 2. 그 이미지의 키포인트/히트맵/PAF 추출 및 렌더링
	// 3. 결과를 디스크에 저장

	// 4. 사용자가 렌더링된 포즈를 표시함
	// 멀티 스레드 시나리오의 모든 것
// 이전의 OpenPose 모듈 외에도, 우리는 다음을 사용할 필요가 있다.
	// 1. `core` module:
		// 포즈 모듈이 필요로 하는 어레이<플로트> 클래스에 대해
		// 스레드 모듈이 큐 사이에 전송하는 데이텀 구조에 대해
	// 2. 기능 모듈: 오류 및 로깅 기능, 즉 op:error & op:log.
//이 파일은 사용자가 특정 예를 취할 때에만 사용해야 한다.

// C++ 스탠다드 라이브러리

#include <string>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <thread> // std::this_thread
#include <cstring> 
#include <iostream>

//boost /timer header include
#include <boost/timer.hpp>

// Other 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif


//CURL HTTP Protocol Lib.
extern "C" {
#include <curl/curl.h>
}

#pragma comment (lib, "wldap32.lib")
#pragma comment (lib, "ws2_32.lib")

// OpenPose dependencies
#include <openpose/headers.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

// "--help" 플래그에 있는 사용 가능한 모든 파라미터 옵션을 보십시오. 예`build/examples/openpose/openpose.bin --help`
// Note: 이 명령은 다른 불필요한 제3자 파일에 대한 플래그를 보여준다. OpenPose의 플래그만 확인
// 실행 가능한 예: 'openpose.bin'의 경우 `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other

DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
	" 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
	" low priority messages and 4 for important ones.");
DEFINE_bool(disable_multi_thread, false, "It would slightly reduce the frame rate in order to highly reduce the lag. Mainly useful"
	" for 1) Cases where it is needed a low latency (e.g. webcam in real-time scenarios with"
	" low-range GPU devices); and 2) Debugging OpenPose when it is crashing to locate the"
	" error.");
DEFINE_int32(profile_speed, 1000, "If PROFILER_ENABLED was set in CMake or Makefile.config files, OpenPose will show some"
	" runtime statistics at this frame number.");
// Producer
DEFINE_int32(camera, 0, "The camera index for cv::VideoCapture. Integer in the range [0, 9]. Select a negative"
	" number (by default), to auto-detect and open the first available camera.");
DEFINE_string(camera_resolution, "-1x-1", "Set the camera resolution (either `--camera` or `--flir_camera`). `-1x-1` will use the"
	" default 1280x720 for `--camera`, or the maximum flir camera resolution available for"
	" `--flir_camera`");
DEFINE_double(camera_fps, 30.0, "Frame rate for the webcam (also used when saving video). Set this value to the minimum"
	" value between the OpenPose displayed speed and the webcam real frame rate.");
DEFINE_string(video, "", "Use a video file instead of the camera. Use `examples/media/video.avi` for our default"
	" example video.");
DEFINE_string(image_dir, "", "Process a directory of images. Use `examples/media/` for our default example folder with 20"
	" images. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_bool(flir_camera, false, "Whether to use FLIR (Point-Grey) stereo camera.");
DEFINE_int32(flir_camera_index, -1, "Select -1 (default) to run on all detected flir cameras at once. Otherwise, select the flir"
	" camera index to run, where 0 corresponds to the detected flir camera with the lowest"
	" serial number, and `n` to the `n`-th lowest serial number camera.");
DEFINE_string(ip_camera, "", "String with the IP camera URL. It supports protocols like RTSP and HTTP.");
DEFINE_uint64(frame_first, 0, "Start on desired frame number. Indexes are 0-based, i.e. the first frame has index 0.");
DEFINE_uint64(frame_last, -1, "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g. if set to"
	" 10, it will process 11 frames (0-10).");
DEFINE_bool(frame_flip, true, "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
DEFINE_int32(frame_rotate, 0, "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat, false, "Repeat frames when finished.");
DEFINE_bool(process_real_time, false, "Enable to keep the original source frame rate (e.g. for video). If the processing time is"
	" too long, it will skip frames. If it is too fast, it will slow it down.");
DEFINE_string(camera_parameter_folder, "models/cameraParameters/flir/", "String with the folder where the camera parameters are located.");
DEFINE_bool(frame_keep_distortion, false, "If false (default), it will undistortionate the image based on the"
	" `camera_parameter_folder` camera parameters; if true, it will not undistortionate, i.e.,"
	" it will leave it as it is.");
// OpenPose
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	" input image resolution.");
DEFINE_int32(num_gpu, -1, "The number of GPU devices to use. If negative, it will use all the available GPUs in your"
	" machine.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_int32(keypoint_scale, 0, "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y)"
	" coordinates that will be saved with the `write_json` & `write_keypoint` flags."
	" Select `0` to scale it to the original source resolution; `1`to scale it to the net output"
	" size (set with `net_resolution`); `2` to scale it to the final output size (set with"
	" `resolution`); `3` to scale it in the range [0,1], where (0,0) would be the top-left"
	" corner of the image, and (1,1) the bottom-right one; and 4 for range [-1,1], where"
	" (-1,-1) would be the top-left corner of the image, and (1,1) the bottom-right one. Non"
	" related with `scale_number` and `scale_gap`.");
DEFINE_int32(number_people_max, 1, "This parameter will limit the maximum number of people detected, by keeping the people with"
	" top scores. The score is based in person area over the image, body part score, as well as"
	" joint score (between each pair of connected body parts). Useful if you know the exact"
	" number of people in the scene, so it can remove false positives (if all the people have"
	" been detected. However, it might also include false negatives by removing very small or"
	" highly occluded people. -1 will keep them all.");
// OpenPose Body Pose
DEFINE_bool(body_disable, false, "Disable body keypoint detection. Option only possible for faster (but less accurate) face"
	" keypoint detection.");
DEFINE_string(model_pose, "BODY_25", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
	"`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(net_resolution, "-1x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
	" decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
	" closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
	" any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
	" input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
	" e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
	" If you want to change the initial scale, you actually want to multiply the"
	" `net_resolution` by your desired initial scale.");
// OpenPose Body Pose Heatmaps and Part Candidates
DEFINE_bool(heatmaps_add_parts, false, "If true, it will fill op::Datum::poseHeatMaps array with the body part heatmaps, and"
	" analogously face & hand heatmaps to op::Datum::faceHeatMaps & op::Datum::handHeatMaps."
	" If more than one `add_heatmaps_X` flag is enabled, it will place then in sequential"
	" memory order: body parts + bkg + PAFs. It will follow the order on"
	" POSE_BODY_PART_MAPPING in `src/openpose/pose/poseParameters.cpp`. Program speed will"
	" considerably decrease. Not required for OpenPose, enable it only if you intend to"
	" explicitly use this information later.");
DEFINE_bool(heatmaps_add_bkg, false, "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
	" background.");
DEFINE_bool(heatmaps_add_PAFs, false, "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
DEFINE_int32(heatmaps_scale, 2, "Set 0 to scale op::Datum::poseHeatMaps in the range [-1,1], 1 for [0,1]; 2 for integer"
	" rounded [0,255]; and 3 for no scaling.");
DEFINE_bool(part_candidates, false, "Also enable `write_json` in order to save this information. If true, it will fill the"
	" op::Datum::poseCandidates array with the body part candidates. Candidates refer to all"
	" the detected body parts, before being assembled into people. Note that the number of"
	" candidates is equal or higher than the number of final body parts (i.e. after being"
	" assembled into people). The empty body parts are filled with 0s. Program speed will"
	" slightly decrease. Not required for OpenPose, enable it only if you intend to explicitly"
	" use this information.");
// OpenPose Face
DEFINE_bool(face, false, "Enables face keypoint detection. It will share some parameters from the body pose, e.g."
	" `model_folder`. Note that this will considerable slow down the performance and increse"
	" the required GPU memory. In addition, the greater number of people on the image, the"
	" slower OpenPose will be.");
DEFINE_string(face_net_resolution, "368x368", "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the face keypoint"
	" detector. 320x320 usually works fine while giving a substantial speed up when multiple"
	" faces on the image.");
// OpenPose Hand
DEFINE_bool(hand, false, "Enables hand keypoint detection. It will share some parameters from the body pose, e.g."
	" `model_folder`. Analogously to `--face`, it will also slow down the performance, increase"
	" the required GPU memory and its speed depends on the number of people.");
DEFINE_string(hand_net_resolution, "368x368", "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the hand keypoint"
	" detector.");
DEFINE_int32(hand_scale_number, 1, "Analogous to `scale_number` but applied to the hand keypoint detector. Our best results"
	" were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4.");
DEFINE_double(hand_scale_range, 0.4, "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range"
	" between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if"
	" scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");
DEFINE_bool(hand_tracking, false, "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate"
	" is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it"
	" simply looks for hands in positions at which hands were located in previous frames, but"
	" it does not guarantee the same person ID among frames.");
// OpenPose 3-D Reconstruction
DEFINE_bool(3d, false, "Running OpenPose 3-D reconstruction demo: 1) Reading from a stereo camera system."
	" 2) Performing 3-D reconstruction from the multiple views. 3) Displaying 3-D reconstruction"
	" results. Note that it will only display 1 person. If multiple people is present, it will"
	" fail.");
DEFINE_int32(3d_min_views, -1, "Minimum number of views required to reconstruct each keypoint. By default (-1), it will"
	" require all the cameras to see the keypoint in order to reconstruct it.");
DEFINE_int32(3d_views, 1, "Complementary option to `--image_dir` or `--video`. OpenPose will read as many images per"
	" iteration, allowing tasks such as stereo camera processing (`--3d`). Note that"
	" `--camera_parameters_folder` must be set. OpenPose must find as many `xml` files in the"
	" parameter folder as this number indicates.");
// Extra algorithms
DEFINE_bool(identification, false, "Experimental, not available yet. Whether to enable people identification across frames.");
DEFINE_int32(tracking, -1, "Experimental, not available yet. Whether to enable people tracking across frames. The"
	" value indicates the number of frames where tracking is run between each OpenPose keypoint"
	" detection. Select -1 (default) to disable it or 0 to run simultaneously OpenPose keypoint"
	" detector and tracking for potentially higher accurary than only OpenPose.");
DEFINE_int32(ik_threads, 0, "Experimental, not available yet. Whether to enable inverse kinematics (IK) from 3-D"
	" keypoints to obtain 3-D joint angles. By default (0 threads), it is disabled. Increasing"
	" the number of threads will increase the speed but also the global system latency.");
// OpenPose Rendering
DEFINE_int32(part_to_show, 0, "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
	" part heat map, 19 for the background heat map, 20 for all the body part heat maps"
	" together, 21 for all the PAFs, 22-40 for each body part pair PAF.");
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
	" background, instead of being rendered into the original image. Related: `part_to_show`,"
	" `alpha_pose`, and `alpha_pose`.");
// OpenPose Rendering Pose
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be"
	" rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	" while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	" more false positives (i.e. wrong detections).");
DEFINE_int32(render_pose, -1, "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
	" (slower but greater functionality, e.g. `alpha_X` flags). If -1, it will pick CPU if"
	" CPU_ONLY is enabled, or GPU if CUDA is enabled. If rendering is enabled, it will render"
	" both `outputData` and `cvOutputData` with the original image and desired body part to be"
	" shown (i.e. keypoints, heat maps or PAFs).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap, 0.7, "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
	" heatmap, 0 will only show the frame. Only valid for GPU rendering.");
// OpenPose Rendering Face
DEFINE_double(face_render_threshold, 0.4, "Analogous to `render_threshold`, but applied to the face keypoints.");
DEFINE_int32(face_render, -1, "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
	" configuration that `render_pose` is using.");
DEFINE_double(face_alpha_pose, 0.6, "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(face_alpha_heatmap, 0.7, "Analogous to `alpha_heatmap` but applied to face.");
// OpenPose Rendering Hand
DEFINE_double(hand_render_threshold, 0.2, "Analogous to `render_threshold`, but applied to the hand keypoints.");
DEFINE_int32(hand_render, -1, "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same"
	" configuration that `render_pose` is using.");
DEFINE_double(hand_alpha_pose, 0.6, "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(hand_alpha_heatmap, 0.7, "Analogous to `alpha_heatmap` but applied to hand.");
// Result Saving
DEFINE_string(write_images, "", "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format, "png", "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV"
	" function cv::imwrite for all compatible extensions.");
DEFINE_string(write_video, "", "Full file path to write rendered frames in motion JPEG video format. It might fail if the"
	" final path does not finish in `.avi`. It internally uses cv::VideoWriter. Flag"
	" `camera_fps` controls FPS.");
DEFINE_string(write_json, "", "Directory to write OpenPose output in JSON format. It includes body, hand, and face pose"
	" keypoints (2-D and 3-D), as well as pose candidates (if `--part_candidates` enabled).");
DEFINE_string(write_coco_json, "", "Full file path to write people pose data with JSON COCO validation format.");
DEFINE_string(write_coco_foot_json, "", "Full file path to write people foot pose data with JSON COCO validation format.");
DEFINE_string(write_heatmaps, "", "Directory to write body pose heatmaps in PNG format. At least 1 `add_heatmaps_X` flag"
	" must be enabled.");
DEFINE_string(write_heatmaps_format, "png", "File extension and format for `write_heatmaps`, analogous to `write_images_format`."
	" For lossless compression, recommended `png` for integer `heatmaps_scale` and `float` for"
	" floating values.");
DEFINE_string(write_keypoint, "", "(Deprecated, use `write_json`) Directory to write the people pose keypoint data. Set format"
	" with `write_keypoint_format`.");
DEFINE_string(write_keypoint_format, "yml", "(Deprecated, use `write_json`) File extension and format for `write_keypoint`: json, xml,"
	" yaml & yml. Json not available for OpenCV < 3.0, use `write_json` instead.");
// Result Saving - Extra Algorithms
DEFINE_string(write_video_adam, "", "Experimental, not available yet. E.g.: `~/Desktop/adamResult.avi`. Flag `camera_fps`"
	" controls FPS.");
DEFINE_string(write_bvh, "", "Experimental, not available yet. E.g.: `~/Desktop/mocapResult.bvh`.");
// UDP communication
//DEFINE_string(udp_host,                 "",             "Experimental, not available yet. IP for UDP communication. E.g., `192.168.0.1`.");
//DEFINE_string(udp_port,                 "8051",         "Experimental, not available yet. Port number for UDP communication.");
// 사용자가 자신의 변수를 필요로 하는 경우 op::Datum structure를 상속하여 추가할 수 있다.
// UserDatum은 op::Datum, just definition에서 상속되기 때문에 OpenPose 래퍼에 의해 직접 사용될 수 있다.
// Wrapper<UserDatum> instead of Wrapper<op::Datum>
struct UserDatum : public op::Datum
{
	bool boolThatUserNeedsForSomeReason;

	UserDatum(const bool boolThatUserNeedsForSomeReason_ = false) :
		boolThatUserNeedsForSomeReason{ boolThatUserNeedsForSomeReason_ }
	{}
};

//keyPoint전역 변수 처리
std::string nose;
std::string neck;
std::string rShoulder;
std::string lShoulder;
std::string rEye;
std::string lEye;
std::string rEar;
std::string lEar;

std::string noseX;
std::string noseY;
std::string neckX;
std::string neckY;
std::string rShoulderX;
std::string rShoulderY;
std::string lShoulderX;
std::string lShoulderY;
std::string rEyeX;
std::string rEyeY;
std::string lEyeX;
std::string lEyeY;
std::string rEarX;
std::string rEarY;
std::string lEarX;
std::string lEarY;

//http 통신 처리 함수
void httpRequestPose(std::string id, std::string pose) {
	CURL *curl;
	CURLcode res;

	std::string json = { "{\"id\":\"" + id + "\",\"pose\":\"" + pose + "\"}" };
	// In windows, this will init the winsock stuff 
	curl_global_init(CURL_GLOBAL_ALL); // 이 옵션은 thread 메모리 공유에 안전하지 않다. 나는 주석처리함

	//get a curl handle 
	curl = curl_easy_init();

	struct curl_slist *list = NULL;

	if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, "220.81.195.81:4000/pose"); //webserver ip 주소와 포트번호, flask 대상 router

		list = curl_slist_append(list, "Accept: application/json"); // Accept 정의 내용 list에 저장 
		list = curl_slist_append(list, "Content-Type: application/json"); // content-type 정의 내용 list에 저장 
		list = curl_slist_append(list, "Charsets: UTF8"); // charsets 정의 내용 list에 저장 

		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list); // content-type 설정

		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L); // 값을 false 하면 에러가 떠서 공식 문서 참고함 
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 1L); // 값을 false 하면 에러가 떠서 공식 문서 참고함

		curl_easy_setopt(curl, CURLOPT_POST, 1L); //POST option
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json.c_str()); //string의 data라는 내용을 전송 할것이다

		//Perform the request, res will get the return code 
		res = curl_easy_perform(curl); // curl 실행 res는 curl 실행후 응답내용이 
		curl_slist_free_all(list); // CURLOPT_HTTPHEADER 와 세트

		// Check for errors 
		if (res != CURLE_OK)
			fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));

		// always cleanup 
		curl_easy_cleanup(curl); // curl_easy_init 과 세트
	}
	curl_global_cleanup(); // curl_global_init 과 세트*/
}

// W-클래스는 템플릿으로 구현하거나 주어진 간단한 클래스로 구현할 수 있다.
// 사용자들이 그가 어떤 종류의 데이터를 줄 사이에 옮길지 알고 있다.
// 이 경우에 우리는 std::shared_ptr의 std::serDatum의 벡터를 가정한다.

// 디렉토리의 모든 jpg 파일을 읽고 반환
class UserOutputClass
{

public:
	bool display(const std::shared_ptr<std::vector<UserDatum>>& datumsPtr)
	{
		// 사용자 표시/저장/기타 처리
			// datum.cvOutputData: 렌더링된 프레임(포즈 또는 히트맵 포함)
			// datum.poseKeypoints:array<float>로 추정된 포즈
		char key = ' ';
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			cv::imshow("SUPOSE, 올바른 자세를 유지합시다. 종료(ESC)", datumsPtr->at(0).cvOutputData);
			// 이미지를 표시하고 최소 1ms 이상 잔다(대개 이미지를 표시하기 위해 최대 5~10ms까지 잔다)

			key = (char)cv::waitKey(1);

		}
		else {}
		//op::log("Nullptr or empty datumsPtr found.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
		return (key == 27);
	}

	void printKeypoints(const std::shared_ptr<std::vector<UserDatum>>& datumsPtr)
	{
		// 포즈 키포인트 사용법
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{

			op::log("\n키포인트값:");
			// 각각 키 포인트 요소들에 접속시킨다
			//상수 변수로 auto 자동 자료형을 적용시키고 실수형 배열 변수에 
			const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
			op::log("사람 관절 포인트값:");
			auto person = 0;
			//for (auto person = 0; person < poseKeypoints.getSize(0); person++)
			//{
				//person : 관찰되는 사람 인덱스 번호
			op::log("대상 " + std::to_string(person + 1) + "번 사람의 " + "(x, y, 신체):");


			for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
			{

				std::string valueToPrint;

				if (bodyPart == 0 || bodyPart == 1 || bodyPart == 2 || bodyPart == 5 || bodyPart == 15 || bodyPart == 16 || bodyPart == 17 || bodyPart == 18)
				{

					noseX = std::to_string(poseKeypoints[{0, 0}]) + "  ";
					noseY = std::to_string(poseKeypoints[{0, 1}]) + "  ";

					neckX = std::to_string(poseKeypoints[{0, 3}]) + "  ";
					neckY = std::to_string(poseKeypoints[{0, 4}]) + "  ";

					rShoulderX = std::to_string(poseKeypoints[{0, 6}]) + "  ";
					rShoulderY = std::to_string(poseKeypoints[{0, 7}]) + "  ";

					lShoulderX = std::to_string(poseKeypoints[{0, 15}]) + "  ";
					lShoulderY = std::to_string(poseKeypoints[{0, 16}]) + "  ";

					rEyeX = std::to_string(poseKeypoints[{0, 45}]) + "  ";
					rEyeY = std::to_string(poseKeypoints[{0, 46}]) + "  ";

					lEyeX = std::to_string(poseKeypoints[{0, 48}]) + "  ";
					lEyeY = std::to_string(poseKeypoints[{0, 49}]) + "  ";

					rEarX = std::to_string(poseKeypoints[{0, 51}]) + "  ";
					rEarY = std::to_string(poseKeypoints[{0, 52}]) + "  ";

					lEarX = std::to_string(poseKeypoints[{0, 54}]) + "  ";
					lEarY = std::to_string(poseKeypoints[{0, 55}]) + "  ";

					for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++)
					{
						valueToPrint += std::to_string(poseKeypoints[{0, bodyPart, xyscore}]) + " ";
					}
				}
				else {
					valueToPrint += " ";
				}

			}
			nose = noseX + " " + noseY + " 코";
			op::log(nose);

			neck = neckX + " " + neckY + " 목";
			op::log(neck);

			rShoulder = rShoulderX + " " + rShoulderY + " 오른쪽 어깨";
			op::log(rShoulder);

			lShoulder = lShoulderX + " " + lShoulderY + " 왼쪽 어깨";
			op::log(lShoulder);

			rEye = rEyeX + " " + rEyeY + " 오른쪽 눈";
			op::log(rEye);

			lEye = lEyeX + " " + lEyeY + " 왼쪽 눈";
			op::log(lEye);

			rEar = rEarX + " " + rEarY + " 오른쪽 귀";
			op::log(rEar);

			lEar = lEarX + " " + lEarY + " 왼쪽 귀";
			op::log(lEar);

			//}
			op::log(" ");
			// Alternative: 문자열 출력
			//op::log("Face keypoints: " + datumsPtr->at(0).faceKeypoints.toString());
			//op::log("Left hand keypoints: " + datumsPtr->at(0).handKeypoints[0].toString());
			//op::log("Right hand keypoints: " + datumsPtr->at(0).handKeypoints[1].toString());
			// Heatmaps
			const auto& poseHeatMaps = datumsPtr->at(0).poseHeatMaps;
			if (!poseHeatMaps.empty())
			{
				//op::log("Pose heatmaps size: [" + std::to_string(poseHeatMaps.getSize(0)) + ", "
				//	+ std::to_string(poseHeatMaps.getSize(1)) + ", "
				//	+ std::to_string(poseHeatMaps.getSize(2)) + "]");
			//	const auto& faceHeatMaps = datumsPtr->at(0).faceHeatMaps;
				//op::log("Face heatmaps size: [" + std::to_string(faceHeatMaps.getSize(0)) + ", "
				//	+ std::to_string(faceHeatMaps.getSize(1)) + ", "
				//	+ std::to_string(faceHeatMaps.getSize(2)) + ", "
				//	+ std::to_string(faceHeatMaps.getSize(3)) + "]");
			//	const auto& handHeatMaps = datumsPtr->at(0).handHeatMaps;
				//op::log("Left hand heatmaps size: [" + std::to_string(handHeatMaps[0].getSize(0)) + ", "
				//	+ std::to_string(handHeatMaps[0].getSize(1)) + ", "
				//	+ std::to_string(handHeatMaps[0].getSize(2)) + ", "
				//	+ std::to_string(handHeatMaps[0].getSize(3)) + "]");
				//op::log("Right hand heatmaps size: [" + std::to_string(handHeatMaps[1].getSize(0)) + ", "
				//	+ std::to_string(handHeatMaps[1].getSize(1)) + ", "
				//	+ std::to_string(handHeatMaps[1].getSize(2)) + ", "
				//	+ std::to_string(handHeatMaps[1].getSize(3)) + "]");
			}
		}
		else {}
		//op::log("Nullptr or empty datumsPtr found.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
	}
};

int Supose()
{
	try
	{
		op::log("신체 측정 시작....", op::Priority::High);
		const auto timerBegin = std::chrono::high_resolution_clock::now();

		// logging_level
		op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
			__LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::Profiler::setDefaultX(FLAGS_profile_speed);

		// 사용자 정의 구성 적용 - 프로그램 변수에 구글 플래그 적용
		// outputSize
		const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
		// faceNetInputSize
		const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
		// handNetInputSize
		const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
		// producerType
		const auto producerSharedPtr = op::flagsToProducer(FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera,
			FLAGS_flir_camera, FLAGS_camera_resolution, FLAGS_camera_fps,
			FLAGS_camera_parameter_folder, !FLAGS_frame_keep_distortion,
			(unsigned int)FLAGS_3d_views, FLAGS_flir_camera_index);
		// poseModel
		const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
		// JSON 저장
		if (!FLAGS_write_keypoint.empty())
			op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
				" Please, use `write_json` instead.", op::Priority::Max);
		// keypointScale
		const auto keypointScale = op::flagsToScaleMode(FLAGS_keypoint_scale);
		// heatmaps to add
		const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
		const auto heatMapScale = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
		// >1 camera view?
		const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
		// Enabling Google Logging
		const bool enableGoogleLogging = true;
		// Logging
		//op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

		// Configure OpenPose
		//op::log("Configuring OpenPose wrapper...", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		op::Wrapper<std::vector<UserDatum>> opWrapper{ op::ThreadManagerMode::AsynchronousOut };

		// 포즈 구성(기본 및 권장 구성에 WrapperStructPose} 사용)
		const op::WrapperStructPose wrapperStructPose{
			!FLAGS_body_disable, netInputSize, outputSize, keypointScale, FLAGS_num_gpu, FLAGS_num_gpu_start,
			FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
			poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
			FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScale, FLAGS_part_candidates,
			(float)FLAGS_render_threshold, FLAGS_number_people_max, enableGoogleLogging };

		// 페이스 구성(op::WrapperStructFace}을(를) 사용하여 비활성화)
		const op::WrapperStructFace wrapperStructFace{
			FLAGS_face, faceNetInputSize, op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
			(float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold };

		// 핸드 구성(op::WrapperStructHand}을(를) 사용하여 비활성화)
		const op::WrapperStructHand wrapperStructHand{
			FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range, FLAGS_hand_tracking,
			op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
			(float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold };
		// 추가 기능 구성(op::WrapperStructExtra}을(를) 사용하여 비활성화)
		const op::WrapperStructExtra wrapperStructExtra{
			FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
		// 프로듀서(기본값을 사용하여 입력을 비활성화함)
		const op::WrapperStructInput wrapperStructInput{
			producerSharedPtr, FLAGS_frame_first, FLAGS_frame_last, FLAGS_process_real_time, FLAGS_frame_flip,
			FLAGS_frame_rotate, FLAGS_frames_repeat };
		// 소비자(모든 출력을 비활성화하려면 커멘드 또는 기본 인수 사용)
		const auto displayMode = op::DisplayMode::NoDisplay;
		const bool guiVerbose = false;
		const bool fullScreen = false;
		const op::WrapperStructOutput wrapperStructOutput{
			displayMode, guiVerbose, fullScreen, FLAGS_write_keypoint,
			op::stringToDataFormat(FLAGS_write_keypoint_format), FLAGS_write_json, FLAGS_write_coco_json,
			FLAGS_write_coco_foot_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
			FLAGS_camera_fps, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_adam,
			FLAGS_write_bvh };
		//, FLAGS_udp_host, FLAGS_udp_port

		// Configure wrapper
		opWrapper.configure(wrapperStructPose, wrapperStructFace, wrapperStructHand, wrapperStructExtra,
			wrapperStructInput, wrapperStructOutput);

		// 단일 스레드 실행으로 설정(디버깅 및 지연 시간 감소)
		if (FLAGS_disable_multi_thread)
			opWrapper.disableMultiThreading();

		op::log("프로그램 구동 중....", op::Priority::High);
		opWrapper.start();

		// User processing
		UserOutputClass userOutputClass;
		bool userWantsToExit = false;

		//set image		
		cv::Mat imgLshoulderDown;
		cv::Mat imgRshoulderDown;
		cv::Mat imgHead;
		cv::Mat imgWaist;

		imgLshoulderDown = cv::imread("D:\\git\\openpose\\bulid_windows\\examples\\tutorial_wrapper\\6_user_asynchronous_output.dir\\Release\\image\\lShDown_.png");
		imgRshoulderDown = cv::imread("D:\\git\\openpose\\bulid_windows\\examples\\tutorial_wrapper\\6_user_asynchronous_output.dir\\Release\\image\\rShDown_.png");
		imgHead = cv::imread("D:\\git\\openpose\\build_windows\\examples\\tutorial_wrapper\\6_user_asynchronous_output.dir\\Release\\image\\head_.png");
		imgWaist = cv::imread("D:\\git\\openpose\\build_windows\\examples\\tutorial_wrapper\\6_user_asynchronous_output.dir\\Release\\image\\waist_.png");

		boost::timer MyTimer;

		while (1)
		{
			// 프레임 띄우기
			std::shared_ptr<std::vector<UserDatum>> datumProcessed;

			if (opWrapper.waitAndPop(datumProcessed))
			{
				userWantsToExit = userOutputClass.display(datumProcessed);
				userOutputClass.printKeypoints(datumProcessed);

				
				//string to int 
				if(!rShoulderY.empty()){
				int subShoulder = stoi(rShoulderY) - stoi(lShoulderY);

				//clac keypoint values for posedata
				if (50 < subShoulder || -50 > subShoulder)
				{
					if (stoi(rShoulderY) < stoi(lShoulderY)) {
						op::log(MyTimer.elapsed());

						if (MyTimer.elapsed() >= 5.0) {

							op::log("5초 지남");
							cv::imshow("SUPOSE", imgLshoulderDown);
							std::string id = "hello";
							std::string pose = "leftShoulder";
							httpRequestPose(id, pose);
							MyTimer.restart();
						}

					}
					else if (stoi(rShoulderY) > stoi(lShoulderY)) {
						op::log(MyTimer.elapsed());

						if (MyTimer.elapsed() >= 5.0) {
							op::log("5초 지남");

							cv::imshow("SUPOSE", imgRshoulderDown);
							std::string id = "hello";
							std::string pose = "rightShoulder";
							httpRequestPose(id, pose);
							MyTimer.restart();
						}
					}
					else {

					}

				}
				else {
					MyTimer.restart();
					op::log("올바른 자세 유지중..");
				}
				}
			}
			else {
				MyTimer.restart();
			}
			//op::log("Processed datum could not be emplaced.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
			if (userWantsToExit == true) {
				break;
			}
		}
		op::log("멈추는 중...\n", op::Priority::High);
		opWrapper.stop();

		//현재 시간 측정
		const auto now = std::chrono::high_resolution_clock::now();

		//총 시간 측정
		const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::seconds>(now - timerBegin).count();
		const auto message = "신체 측정을 종료합니다. 측정 시간 : "
			+ std::to_string((int)totalTimeSec / 60) + " 분. \n\n";
		op::log(message, op::Priority::High);

		// 성공 메세지 출력 반환
		return 0;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return -1;
	}

}

int main(int argc, char *argv[])
{

	// 명령줄 플래그 파싱
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// 오픈포즈 시작
	return Supose();

}

