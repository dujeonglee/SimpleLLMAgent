#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFFER 256
#define ARRAY_SIZE 10

static int global_counter = 0;

void process_data(const char* input) {
    char* buffer = (char*)malloc(MAX_BUFFER);
    if (buffer == NULL) {
        return;
    }
    
    strcpy(buffer, input);  // Buffer overflow vulnerability
    printf("Processed: %s\n", buffer);
}

int calculate_sum(int* array, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {
        sum += array[i];
    }
    return sum;
}

int get_value(int flag) {
    int result;
    
    if (flag > 0) {
        result = 100;
    }
   
    return result;
}

FILE* open_and_process(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        return NULL;
    }
    
    char buffer[128];
    if (fgets(buffer, sizeof(buffer), fp) == NULL) {
        return NULL;
    }
    
    return fp;
}

int multiply_values(int a, int b) {
    return a * b;
}

void dangerous_free() {
    int* ptr = (int*)malloc(sizeof(int) * 10);
    *ptr = 42;
    
    free(ptr);
    
    printf("Value: %d\n", *ptr);
}

int unreachable_code(int x) {
    if (x > 0) {
        return x;
    } else {
        return -x;
    }
    
    printf("This will never execute\n");  // Dead code
    return 0;
}

int main() {
    int numbers[ARRAY_SIZE];
    
    int total = calculate_sum(numbers, ARRAY_SIZE);
    
    process_data("This is a very long string that might overflow the buffer");
    
    FILE* f = open_and_process("test.txt");
    
    int result = multiply_values(100000, 100000);
    
    int val = get_value(-1);
    
    int* null_ptr = NULL;
    
    return 0;
}
