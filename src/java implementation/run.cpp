#include <jni.h>
#include <jni_md.h>
#include <iostream>

//g++ -std=c++11 -o2 run.cpp -I/Library/Java/JavaVirtualMachines/jdk1.8.0_102.jdk/Contents/Home/include/ \
 -I/Library/Java/JavaVirtualMachines/jdk1.8.0_102.jdk/Contents/Home/include/darwin/ \
 -o run -lm

//g++ -o run \
 -I/Library/Java/JavaVirtualMachines/jdk1.8.0_102.jdk/Contents/Home/include \
 -I/Library/Java/JavaVirtualMachines/jdk1.8.0_102.jdk/Contents/Home/include/darwin \
 -L/Library/Java/JavaVirtualMachines/jdk1.8.0_102.jdk/Contents/Home/jre/lib/server \
 run.cpp \
 -ljvm
//export LD_LIBRARY_PATH=/Library/Java/JavaVirtualMachines/jdk1.8.0_102.jdk/Contents/Home/jre/lib/server/


// using namespace std;
// int main() {
//   JavaVMOption options[1];
//   JNIEnv *env;
//   JavaVM *jvm;
//   JavaVMInitArgs vm_args;
//   long status;
//   jclass cls;
//   jmethodID mid;

//   options[0].optionString = "-Djava.class.path=.:./";
//   memset(&vm_args, 0, sizeof(vm_args));
//   vm_args.version = JNI_VERSION_1_2;
//   vm_args.nOptions = 1;
//   vm_args.options = options;
//   status = JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);

//   if(status != JNI_ERR) {
//     cls = env->FindClass("Main");
//     if(cls != 0) {
//       jmethodID mid = env->GetStaticMethodID(cls, "main", "()V");
//       if(mid == nullptr) {
//         cerr << "ERROR: method void main not found !" << endl;
//       } else {
//         env->CallStaticVoidMethod(cls, mid);
//         cout << endl;
//       }      
//     }
//     jvm->DestroyJavaVM();
//     return 0;
//   } 
//   else {
//     return -1;
//   }

// }

using namespace std;
int main() {
    // Using namespace std;
    JavaVM *jvm;
    JNIEnv *env;

    JavaVMInitArgs vm_args;
    JavaVMOption* options = new JavaVMOption[1];
    options[0].optionString = "-Djava.class.path=.";
    vm_args.version = JNI_VERSION_1_8;
    vm_args.nOptions = 1;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = false;

    JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);
    delete options;
    // if (rc != JNI_OK) {
    //     return -1;
    // }

    cout << "JVM load succeeded: Version ";
    jint ver = env->GetVersion();
    cout << ((ver >> 16)&0x0f) << "."<<(ver&0x0f) << endl;

    // TO DO: add the code that will use JVM <============  (see next steps)
    jclass cls1 = env->FindClass("go/Main");

    jmethodID constructor;
    jobject instance;

    if(cls1 == 0) {
        cerr << "ERROR: class not found\n";
    }
    else {
        cout << "Class Main found" << endl;
        constructor = env->GetMethodID(cls1, "<init>", "()V");
        if(constructor != 0) {
            cout << "Constructor found\n";
            jobject instance = env->NewObject(cls1, constructor);
            jmethodID mid = env->GetStaticMethodID(cls1, "init", "()V");
            if(mid == 0) {
                cerr << "ERROR: method void init not found !" << endl;
            } else {
                if(instance != 0) {
                    cout << "Instance created\n";
                    env->CallObjectMethod(instance, mid);
                }
            }
        }
        // jmethodID mid = env->GetStaticMethodID(cls1, "mymain", "()V");
        // if(mid == 0) {
        //     cerr << "ERROR: method void init not found !" << endl;
        // } else {
        //     env->CallStaticVoidMethod(cls1, mid);
        //     cout << endl;
        // }
    }

    jvm->DestroyJavaVM();
    return 0;

}
