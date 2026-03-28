-- NOT MY PROJECT --


  
  
---

   
# VulkanHello Android App

This is a production-ready Android application that demonstrates Vulkan rendering of a rotating tetrahedron.

## Prerequisites

1. Install Android Studio with NDK and CMake.
2. Create a new project: **File > New > New Project > Native C++**.
3. Select **Kotlin** as the language.

## Features

- Vulkan-based 3D rendering
- Rotating tetrahedron with colored vertices
- Android SurfaceView integration
- GLM math library for transformations

## Build and Run

1. Open the project in Android Studio.
2. Ensure NDK and CMake are installed.
3. Build and run on a device with Vulkan support (API 24+).

## Project Structure

- `app/src/main/AndroidManifest.xml` - App manifest
- `app/src/main/java/com/example/vulkanhello/MainActivity.kt` - Main activity
- `app/src/main/cpp/main.cpp` - Vulkan initialization and rendering
- `app/src/main/cpp/CMakeLists.txt` - Build configuration
- `app/src/main/assets/shaders/` - GLSL shaders
- `app/src/main/res/layout/activity_main.xml` - UI layout

## Authors

Воронцов Илья, Филиппов Кирилл 
