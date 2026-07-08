#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef void (*RKLLMCallback)(void *data, void *user_data, int state, int len_val);

typedef struct {
    char padding[128];
} RKLLMParam;

static void *g_lib = NULL;
static void *g_handle = NULL;
static int g_initialized = 0;

typedef RKLLMParam (*FnCreateDefaultParam)(void);
typedef int (*FnInit)(void **handle, RKLLMParam *param, const char *model_path,
                      RKLLMCallback callback, int max_new_tokens, int max_context_len);
typedef int (*FnRun)(void **handle, const char *prompt, void *custom, int len);
typedef int (*FnDestroy)(void **handle);
typedef int (*FnAbort)(void **handle);

static FnCreateDefaultParam fn_create_default_param = NULL;
static FnInit fn_init = NULL;
static FnRun fn_run = NULL;
static FnDestroy fn_destroy = NULL;
static FnAbort fn_abort = NULL;

static RKLLMCallback g_user_callback = NULL;

static void internal_callback(void *data, void *user_data, int state, int len_val) {
    if (g_user_callback) {
        g_user_callback(data, user_data, state, len_val);
    }
}

int wrapper_load_library(const char *lib_path) {
    if (g_lib) {
        dlclose(g_lib);
        g_lib = NULL;
    }

    g_lib = dlopen(lib_path, RTLD_NOW);
    if (!g_lib) {
        return -1;
    }

    fn_create_default_param = (FnCreateDefaultParam)dlsym(g_lib, "rkllm_createDefaultParam");
    fn_init = (FnInit)dlsym(g_lib, "rkllm_init");
    fn_run = (FnRun)dlsym(g_lib, "rkllm_run");
    fn_destroy = (FnDestroy)dlsym(g_lib, "rkllm_destroy");
    fn_abort = (FnAbort)dlsym(g_lib, "rkllm_abort");

    if (!fn_create_default_param || !fn_init || !fn_run || !fn_destroy) {
        dlclose(g_lib);
        g_lib = NULL;
        return -2;
    }

    return 0;
}

int wrapper_init(const char *model_path, RKLLMCallback callback,
                 int max_new_tokens, int max_context_len) {
    if (!fn_create_default_param || !fn_init) {
        return -1;
    }

    g_handle = NULL;

    RKLLMParam param = fn_create_default_param();

    g_user_callback = callback;

    int ret = fn_init(&g_handle, &param, model_path, internal_callback,
                      max_new_tokens, max_context_len);
    if (ret != 0) {
        return ret;
    }

    g_initialized = 1;
    return 0;
}

int wrapper_run(const char *prompt) {
    if (!fn_run || !g_initialized) {
        return -1;
    }
    return fn_run(&g_handle, prompt, NULL, 0);
}

int wrapper_destroy(void) {
    if (!fn_destroy || !g_initialized) {
        return -1;
    }
    int ret = fn_destroy(&g_handle);
    g_initialized = 0;
    g_handle = NULL;
    return ret;
}

int wrapper_abort(void) {
    if (!fn_abort || !g_initialized) {
        return -1;
    }
    return fn_abort(&g_handle);
}
