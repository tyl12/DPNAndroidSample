#include <jni.h>
#include <time.h>
#include <unistd.h>


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stddef.h>

#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/inotify.h>
#include <dirent.h>
#include <fcntl.h>
#include <errno.h>
#include <ctype.h>

#include "dp_api.h"
#include "Fp16Convert.h"
#include "mv_types.h"
#include "interpret_output.h"
#include "Common.h"
#include "Region.h"

#include <android/log.h>
#include <array>
#include <dirent.h>
#include <fstream>

#define LOG_TAG     "Deepano"
#define ALOGE(...)      __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

// jni extern
EXTERN {
JNIEXPORT jint JNICALL
Java_com_deepano_dpnandroidsample_DeepanoApiFactory_initDevice(
        JNIEnv *env,
        jobject /* this */, jint fd);

JNIEXPORT jint JNICALL
Java_com_deepano_dpnandroidsample_DeepanoApiFactory_startCamera(
        JNIEnv *env,
        jobject /* this */);

JNIEXPORT jint JNICALL
Java_com_deepano_dpnandroidsample_DeepanoApiFactory_netProc(
        JNIEnv *env,
        jobject /* this */, jstring blobPath);
};

#if 1
#define MOVIDIUS_FP32

extern "C" {


unsigned int rnd_mode;
unsigned int exceptionsReg;
unsigned int *exceptions = &exceptionsReg;
unsigned int f16_shift_left(unsigned int op, unsigned int cnt) {
    unsigned int result;
    if (cnt == 0) {
        result = op;
    } else if (cnt < 32) {
        result = (op << cnt);
    } else {
        result = 0;
    }
    return result;
}

float f16Tof32(unsigned int x) {
    unsigned int sign;
    int exp;
    unsigned int frac;
    unsigned int result;
    u32f32 u;

    frac = EXTRACT_F16_FRAC(x);
    exp = EXTRACT_F16_EXP(x);
    sign = EXTRACT_F16_SIGN(x);
    if (exp == 0x1F) {
        if (frac != 0) {
            // NaN
            if (F16_IS_SNAN(x)) {
                *exceptions |= F32_EX_INVALID;
            }
            result = 0;
            //Get rid of exponent and sign
#ifndef MOVIDIUS_FP32
            result = x << 22;
              result = f32_shift_right(result, 9);
              result |= ((sign << 31) | 0x7F800000);
#else
            result |= ((sign << 31) | 0x7FC00000);
#endif
        } else {
            //infinity
            result = PACK_F32(sign, 0xFF, 0);
        }
    } else if (exp == 0) {
        //either denormal or zero
        if (frac == 0) {
            //zero
            result = PACK_F32(sign, 0, 0);
        } else {
            //subnormal
#ifndef MOVIDIUS_FP32
            f16_normalize_subnormal(&frac, &exp);
              exp--;
              // ALDo: is the value 13 ok??
              result = f16_shift_left(frac, 13);
              // exp = exp + 127 - 15 = exp + 112
              result = PACK_F32(sign, (exp + 0x70), result);
#else
            result = PACK_F32(sign, 0, 0);
#endif
        }
    } else {
        // ALDo: is the value 13 ok??
        result = f16_shift_left(frac, 13);
        result = PACK_F32(sign, (exp + 0x70), result);
    }

    u.u32 = result;
    return u.f32; //andreil
}


} //extern "C"

#endif
//typedef enum NET_CAFFE_TENSFLOW {
//    DP_AGE_NET = 0,
//    DP_ALEX_NET,
//    DP_GOOGLE_NET,
//    DP_GENDER_NET,
//    DP_TINI_YOLO_NET,
//    DP_SSD_MOBILI_NET,
//    DP_RES_NET,
//    DP_SQUEEZE_NET,
//    DP_MNIST_NET,
//    DP_INCEPTION_V1,
//    DP_INCEPTION_V2,
//    DP_INCEPTION_V3,
//    DP_INCEPTION_V4,
//    DP_MOBILINERS_NET,
//    DP_ALI_FACENET,
//    DP_TINY_YOLO_V2_NET,
//    DP_FACE_NET,
//    DP_CAFFE_Nmd
//} DP_MODEL_NET;


typedef enum NET_CAFFE_TENSFLOW
{
    DP_AGE_NET=0,
    DP_ALEX_NET,
    DP_GOOGLE_NET,
    DP_GENDER_NET,
    DP_TINI_YOLO_NET,
    DP_SSD_MOBILI_NET,
    DP_RES_NET,
    DP_SQUEEZE_NET,
    DP_MNIST_NET,
    DP_INCEPTION_V1,
    DP_MOBILINERS_NET,
    DP_TINY_YOLO_V2_NET,
    DP_TINY_YOLO_V2_FACE_NET,
    DP_FACE_NET,
} DP_MODEL_NET;


//
JavaVM *g_VM;
jobject g_obj;
jclass g_coordBoxClass;

jint devStatus = -1;
dp_image_box_t BLOB_IMAGE_SIZE = {0, 1280, 0, 960};
dp_image_box_t box_demo[100];
char categoles[100][20];
int num_box_demo = 0;

// common extern
extern void video_callback(dp_img_t *img, void *param);

extern void box_callback_model_demo(void *result, void *param);


#define PEOPLE_CNT (32)
float VALID_PEOPLE[PEOPLE_CNT][128];
char PEOPLE_NAME[PEOPLE_CNT][32];

float VALID_PEOPLE_FOUR_STAGE[PEOPLE_CNT][4];

int Total_valid_people = 0;
float resultfp32[128] = {0};
float result_stage[4] = {0};

int load_local_faces()
{
    DIR  *dir;
    struct dirent *ptr;
    dir = opendir("/sdcard/face");
    if (dir == NULL){
        ALOGE("no face dir found");
        return -1;
    }

    int index=0;
    while((ptr = readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
            continue;
        char filepath[30]="/sdcard/face/";

        strcat(filepath,ptr->d_name);
        printf("filepath:%s\n",filepath);
        ALOGE("filepath:%s\n",filepath);

        std::ifstream f(filepath);
        std::string s;
        int cnt = 0;
        while(getline(f, s)){
            float val = atof(s.c_str());
            VALID_PEOPLE[index][cnt++] = val;
            if (cnt == 128)
                break;
        }

        memcpy(PEOPLE_NAME[index],ptr->d_name,32);
        std::cout<<"read people feature: "<<(ptr->d_name)<<std::endl;
        ALOGE("read people feature: %s\n", (ptr->d_name));

        if (index++ == 32)
            break;
    }

#if 0
    for(int ii=0;ii<index;ii++)
        {
            for(int jj=0;jj<128;jj++)
            {
                printf(" %.3f  ",VALID_PEOPLE[ii][jj]);
                VALID_PEOPLE_SUM_SIGNAL[ii]+=VALID_PEOPLE[ii][jj];
                if(ii<32)
                    VALID_PEOPLE_FOUR_STAGE[ii][0]+=VALID_PEOPLE[ii][jj];
                else if(ii<64)
                    VALID_PEOPLE_FOUR_STAGE[ii][1]+=VALID_PEOPLE[ii][jj];
                else if(ii<96)
                    VALID_PEOPLE_FOUR_STAGE[ii][2]+=VALID_PEOPLE[ii][jj];
                else if(ii<128)
                    VALID_PEOPLE_FOUR_STAGE[ii][3]+=VALID_PEOPLE[ii][jj];
            }

            printf("\n");
        }
#endif
    Total_valid_people=index;
    printf("Total_valid_people:%d\n",Total_valid_people);
    ALOGE("Total_valid_people:%d\n",Total_valid_people);
    closedir(dir);
    return 0;
}

std::array<float, 128> print_facenet_result(void *fathomOutput,char *peo_name)
{
    ALOGE("%s E", __FUNCTION__);
    std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
    for(int ii=0;ii<4;ii++)
        result_stage[ii]=0.0f;

    u16* probabilities=(u16*)fathomOutput;

    unsigned int resultlen=128;

    float tmp_diff_sum=0.0f;
    float tmp_diff_stage=0.0f;
    float tmp_diff_squre=0.0f;

    float Total_diff[32];
    float this_diff=0.0f;
    float tmp_mini_diff=100.0f;//set a large one by default
    int tmp_mini_index=0;
    std::array<float, 128> feature={0};
    for(u32 i=0;i<resultlen;i++)
    {

        resultfp32[i]=f16Tof32(probabilities[i]);
        feature[i]=resultfp32[i];
        //printf("resultfp32[%d]:%.3f\n",i,resultfp32[i]);
        //ALOGE("resultfp32[%d]:%.3f",i, resultfp32[i]);


        if(i<32)
            result_stage[0]+=resultfp32[i];
        if(i>=32&&i<64)
            result_stage[1]+=resultfp32[i];
        if(i>=64&&i<96)
            result_stage[2]+=resultfp32[i];
        if(i>=96&&i<128)
            result_stage[3]+=resultfp32[i];
    }
    for(int ii=0;ii<4;ii++){
        ALOGE("result_stage[%d]:%.3f ",ii,result_stage[ii]);
        printf("result_stage[%d]:%.3f\n",ii,result_stage[ii]);
    }
    ALOGE("\n");

    for(int valid_people_index=0;valid_people_index<Total_valid_people;valid_people_index++)
    {
        float diff=0.0f;
        tmp_diff_squre=0.0f;
        tmp_diff_stage=0.0f;
        for(int output_index=0;output_index<resultlen;output_index++)
        {
            diff=VALID_PEOPLE[valid_people_index][output_index]-resultfp32[output_index];
            //this_diff=pow(diff,2.0f);
            this_diff= diff * diff;
            tmp_diff_squre+=this_diff;
        }
        ALOGE("tmp_diff_squre:%.5f \n",tmp_diff_squre);
        /*for(int ii=0;ii<4;ii++)
          {
          diff=result_stage[ii]-VALID_PEOPLE_FOUR_STAGE[valid_people_index][ii];
          tmp_diff_stage+=pow(diff,2.0f);
          }
          printf("\ntmp_diff_stage:%.3f \n",tmp_diff_stage);*/
        Total_diff[valid_people_index]=tmp_diff_squre;

        ALOGE("\nTotal_diff[%d]:%.5f \n",valid_people_index,Total_diff[valid_people_index]);
        if(valid_people_index==0)
        {
            tmp_mini_diff=Total_diff[0];
            tmp_mini_index=0;
        }
        else
        {
            if(tmp_mini_diff>Total_diff[valid_people_index])
            {
                tmp_mini_diff=Total_diff[valid_people_index];
                tmp_mini_index=valid_people_index;
            }
        }
    }
    if (Total_valid_people > 0){
        ALOGE("\nmin diff:%.3f\n",tmp_mini_diff);
        printf("\nmin diff:%.3f\n",tmp_mini_diff);

        if(tmp_mini_diff>=-1 && tmp_mini_diff<= 1.0)
            memcpy(peo_name,PEOPLE_NAME[tmp_mini_index],32);

        ALOGE("detectored_name:%s\n",peo_name);
        printf("detectored_name:%s\n",peo_name);
    }

    for(int ii=0;ii<4;ii++)
        result_stage[ii]=0.0f;

    return feature;
}

JNIEXPORT jint JNICALL
Java_com_deepano_dpnandroidsample_DeepanoApiFactory_initDevice(
        JNIEnv *env,
        jobject thiz, jint fd) {
    int ret;

    env->GetJavaVM(&g_VM);
    g_obj = env->NewGlobalRef(thiz);

    jclass coordBoxClass = env->FindClass("com/deepano/dpnandroidsample/CoordBox");
    g_coordBoxClass = static_cast<jclass>(env->NewGlobalRef(coordBoxClass));


    load_local_faces();
    ret = dp_init(fd);
    if (ret == 0) {
        devStatus = 0;
        ALOGE("init device successfully\n");
    } else {
        ALOGE("init device failed\n");
    }
    return ret;

}

JNIEXPORT jint JNICALL
Java_com_deepano_dpnandroidsample_DeepanoApiFactory_startCamera(
        JNIEnv *env,
        jobject thiz) {
    if (devStatus == -1) {
        ALOGE("Please init device first!\n");
        return -1;
    }
    int ret;
    int param = 15;

    dp_register_video_frame_cb(video_callback, &param);
    ret = dp_start_camera_video();
    if (ret == 0) {
        ALOGE("start video successfully\n");
    } else {
        ALOGE("start video failed!\n");
        return -1;
    }
    return 0;

}

DP_MODEL_NET net_1 = DP_SSD_MOBILI_NET;
DP_MODEL_NET net_2=DP_FACE_NET;

JNIEXPORT jint JNICALL
Java_com_deepano_dpnandroidsample_DeepanoApiFactory_netProc(
        JNIEnv *env,
        jobject /* this */, jstring blobPath) {
    jint ret;
//    jint blob_nums = 1; //numbers of the blobs；
//    dp_blob_parm_t parms = {0, 300, 300, 707 * 2}; // NN image input size
//    dp_netMean mean = {127.5, 127.5, 127.5, 127.5}; //average && std

    jint blob_nums = 2; //numbers of the blobs；
    dp_blob_parm_t parms[2] = {
            {0, 300, 300, 707 * 2},
            {0,160,160,128*2}}; // NN image input size
    dp_netMean mean[2]={{0,0,0,255},{112.2917,112.2917,112.2917,59.7970}};

    const char *path = env->GetStringUTFChars(blobPath, 0);
    ALOGE("Blob Path = %s\n", path);

    dp_set_blob_image_size(&BLOB_IMAGE_SIZE); //here is 1280*960;it can be modified to a customization size
    dp_set_blob_parms(blob_nums, parms); // transfer blob params
    dp_set_blob_mean_std(blob_nums, mean); //transfer average && std

    ret = dp_update_model(path); // transfer blob model
    if (ret == 0) {
        ALOGE("Test dp_update_model1 sucessfully!\n");
    } else {
        ALOGE("Test dp_update_model1 failed !\n");
        return -1;
    }

    ret = dp_update_model_2("/sdcard/face_recogntion.graph"); // transfer blob model
    if (ret == 0) {
        ALOGE("Test dp_update_model2 sucessfully!\n");
    } else {
        ALOGE("Test dp_update_model2 failed !\n");
        return -1;
    }

    dp_register_box_device_cb(box_callback_model_demo, &net_1); //receive the output buffer of the first NN
    //dp_register_fps_device_cb(fps_callback,&net); // fps
    //dp_register_parse_blob_time_device_cb(blob_parse_callback,NULL); // model parsing-time

    dp_register_second_box_device_cb(box_callback_model_demo,&net_2);

    dp_register_video_frame_cb(video_callback, &net_1); // video callback

    ret = dp_start_camera_video(); // NN will start to work if camera is on
    if (ret == 0) {
        ALOGE("Test test_start_video successfully!\n");
    } else {
        ALOGE("Test test_start_video failed! ret=%d\n", ret);
        return -1;
    }

    env->ReleaseStringUTFChars(blobPath, path);
    return 0;
}


void video_callback(dp_img_t *img, void *param) {

    JNIEnv *env;

    int getEnvStat = g_VM->GetEnv((void **) &env, JNI_VERSION_1_6);
    if (getEnvStat == JNI_EDETACHED) {
        if (g_VM->AttachCurrentThread(&env, NULL) != 0) {
            return;
        }
    }

    jclass javaClass = env->GetObjectClass(g_obj);

    if (javaClass == 0) {
        ALOGE("g_class is null\n");
        g_VM->DetachCurrentThread();
        return;
    }

    jmethodID javaCallbackId = env->GetMethodID(javaClass, "update", "([B)V");
    if (javaCallbackId == 0) {
        ALOGE("javaCallbackId is 0\n");
        return;
    }

    jbyteArray yuvBuffer = env->NewByteArray(1280 * 960 * 3 / 2);
    env->SetByteArrayRegion(yuvBuffer, 0, 1280 * 960 * 3 / 2,
                            reinterpret_cast<const jbyte *>(img->img));
    env->CallVoidMethod(g_obj, javaCallbackId, yuvBuffer);
    env->DeleteLocalRef(yuvBuffer);
    env->DeleteLocalRef(javaClass);
}


int dump_index = 0;
int num_valid_boxes = 0;
int ProcessedBoxCnt=0;
std::vector<std::string> nameVector;

//int type = 0;
void box_callback_model_demo(void *result, void *param) {
    DP_MODEL_NET model = *((DP_MODEL_NET *) param);
    // i have a bug here,can not fetch the right param,
    // this is a movidius system bug,i will fix it later.

    //if (type == 0){
    if( model == DP_SSD_MOBILI_NET) {
        ALOGE("cdk_result_model: DP_SSD_MOBILI_NET");
        char *category[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                            "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                            "train", "tvmonitor"};
        u16 *probabilities = (u16 *) result;
        unsigned int resultlen = 707;
        float *resultfp32;
        resultfp32 = (float *) malloc(resultlen * sizeof(*resultfp32));
        int img_width = 1280;
        int img_height = 960;
        for (u32 i = 0; i < resultlen; i++)
            resultfp32[i] = f16Tof32(probabilities[i]);
        num_valid_boxes = int(resultfp32[0]);
//        if (num_valid_boxes >5 || num_box_demo < 0){
//            type = 0;
//            free(resultfp32);
//            return;
//        }
        int index = 0;
        ALOGE("num_valid_bxes:%d\n", num_valid_boxes);
        for (int box_index = 0; box_index < num_valid_boxes; box_index++) {
            int base_index = 7 * box_index + 7;
            if (resultfp32[base_index + 6] < 0
                || resultfp32[base_index + 6] >= 1
                || resultfp32[base_index + 5] < 0
                || resultfp32[base_index + 5] >= 1
                || resultfp32[base_index + 4] < 0
                || resultfp32[base_index + 4] >= 1
                || resultfp32[base_index + 3] < 0
                || resultfp32[base_index + 3] >= 1
                || resultfp32[base_index + 2] >= 1
                || resultfp32[base_index + 2] < 0
                || resultfp32[base_index + 1] < 0) {
                continue;
            }
            ALOGE(":::::%d %f %f %f %f %f\n",
                  int(resultfp32[base_index + 1]),
                  resultfp32[base_index + 2],
                  resultfp32[base_index + 3],
                  resultfp32[base_index + 4],
                  resultfp32[base_index + 5],
                  resultfp32[base_index + 6]);
            box_demo[index].x1 = (int(resultfp32[base_index + 3] * img_width) > 0) ? int(
                    resultfp32[base_index + 3] * img_width) : 0;
            box_demo[index].x2 = (int(resultfp32[base_index + 5] * img_width) < img_width) ? int(
                    resultfp32[base_index + 5] * img_width) : img_width;
            box_demo[index].y1 = (int(resultfp32[base_index + 4] * img_height) > 0) ? int(
                    resultfp32[base_index + 4] * img_height) : 0;
            box_demo[index].y2 = (int(resultfp32[base_index + 6] * img_height) < img_height) ? int(
                    resultfp32[base_index + 6] * img_height) : img_height;
            memcpy(categoles[index], category[int(resultfp32[base_index + 1])], 20);
            index++;
        }
        //usleep(2000 * 1000);
        num_box_demo = index;
        free(resultfp32);
        ProcessedBoxCnt = 0;
        nameVector.clear();
        if (num_box_demo > 0) {
            nameVector.reserve(num_box_demo);
            //type = 1;
        }
    }
    else if (model == DP_FACE_NET)
    //else if (type == 1)
    {
        ALOGE("cdk_result_model: DP_FACE_NET");
        char detector_people_name[32]={'\n'};
        std::array<float, 128> feature = print_facenet_result(result,detector_people_name);

        {
            char fname[64];
            sprintf(fname, "/sdcard/feature_%d_%s.txt", dump_index, detector_people_name);
            if (strlen(detector_people_name) > 1)
                ALOGE("Found face: %s", detector_people_name);
            std::ofstream of(fname);
            for (u32 i = 0; i < feature.size(); i++) {
                //of << i<<": "<< feature[i]<<std::endl;
                of << feature[i] << std::endl;
            }
        }
        nameVector.push_back(detector_people_name);

        ProcessedBoxCnt++;
        if (ProcessedBoxCnt == num_box_demo)
        {
            //type = 0;
            JNIEnv *env;

            int getEnvStat = g_VM->GetEnv((void **) &env, JNI_VERSION_1_6);
            if (getEnvStat == JNI_EDETACHED) {
                if (g_VM->AttachCurrentThread(&env, NULL) != 0) {
                    return;
                }
            }

            jclass javaClass = env->GetObjectClass(g_obj);

            if (javaClass == 0) {
                ALOGE("g_class is null\n");
                g_VM->DetachCurrentThread();
                return;
            }
            jmethodID javaCallbackId = env->GetMethodID(javaClass, "getCoordinate",
                                                        "([Lcom/deepano/dpnandroidsample/CoordBox;)V");

            if (javaCallbackId == 0) {
                ALOGE("javaCallbackId is 0\n");
                return;
            }

            jobjectArray boxArray;
            boxArray = env->NewObjectArray(num_valid_boxes, g_coordBoxClass, 0);

            jfieldID x1 = env->GetFieldID(g_coordBoxClass, "x1", "I");
            jfieldID y1 = env->GetFieldID(g_coordBoxClass, "y1", "I");
            jfieldID x2 = env->GetFieldID(g_coordBoxClass, "x2", "I");
            jfieldID y2 = env->GetFieldID(g_coordBoxClass, "y2", "I");
            jfieldID name = env->GetFieldID(g_coordBoxClass, "name", "Ljava/lang/String;");

            jmethodID objectClassInitID = (env)->GetMethodID(g_coordBoxClass, "<init>", "()V");
            jobject objectNewEng;
            for (int box_index = 0; box_index < num_valid_boxes; box_index++) {
                objectNewEng = env->NewObject(g_coordBoxClass, objectClassInitID);
                env->SetIntField(objectNewEng, x1, box_demo[box_index].x1);
                env->SetIntField(objectNewEng, y1, box_demo[box_index].y1);
                env->SetIntField(objectNewEng, x2, box_demo[box_index].x2);
                env->SetIntField(objectNewEng, y2, box_demo[box_index].y2);

                jstring j_str = env->NewStringUTF(nameVector[box_index].c_str());
                env->SetObjectField(objectNewEng, name, j_str);

                env->SetObjectArrayElement(boxArray, box_index, objectNewEng);
                env->DeleteLocalRef(objectNewEng);
            }

            env->CallVoidMethod(g_obj, javaCallbackId, boxArray);
            env->DeleteLocalRef(boxArray);
            env->DeleteLocalRef(javaClass);
        }
    }
}
