#include <tvm/runtime/module.h>

#include <iostream>
#include <string>
#include <dlfcn.h>

int main() {
    std::string name = "./libLlama-2-7b-chat-hf-q4f16_1-android.so";
    std::cout << "DLSym output: " << dlsym(nullptr, "TVMFuncCall") << std::endl;
    void* lib_handle = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (lib_handle == nullptr) {
        std::cout << "Cannot open library: " << name
                  << "\nError: " << dlerror()
                  << std::endl;
    } else {
        dlclose(lib_handle);
    }
    auto xecutable = tvm::runtime::Module::LoadFromFile(name);
    return 0;
}
