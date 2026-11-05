#import <Cocoa/Cocoa.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        NSString *payload = [[[NSBundle mainBundle] resourcePath] stringByAppendingPathComponent:@"payload"];
        NSString *python = [payload stringByAppendingPathComponent:@"runtime/bin/python3"];
        NSString *script = [payload stringByAppendingPathComponent:@"bootstrap.py"];
        unsetenv("PYTHONHOME"); unsetenv("PYTHONPATH");
        char **args = calloc((size_t)argc + 4, sizeof(char *));
        args[0] = (char *)[python fileSystemRepresentation];
        args[1] = "-s"; args[2] = "-B"; args[3] = (char *)[script fileSystemRepresentation];
        for (int i = 1; i < argc; ++i) args[i + 3] = argv[i];
        execv(args[0], args);
        NSAlert *alert = [[NSAlert alloc] init];
        alert.messageText = @"AutoGrowth could not start";
        alert.informativeText = @"The bundled Python runtime could not be launched. Reinstall the complete application.";
        [alert runModal];
        free(args);
        return 1;
    }
}
