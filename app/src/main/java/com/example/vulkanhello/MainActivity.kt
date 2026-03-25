package com.example.vulkanhello

import android.app.Activity
import android.os.Bundle
import android.view.SurfaceView
import android.view.SurfaceHolder

class MainActivity : Activity(), SurfaceHolder.Callback {
    
    companion object {
        init {
            System.loadLibrary("vulkanhello")
        }
    }

    private lateinit var surfaceView: SurfaceView
    
    // Native function declarations
    external fun initVulkan(surface: android.view.Surface, assetManager: android.content.res.AssetManager, width: Int, height: Int)
    external fun destroyVulkan()
    external fun resizeVulkan(width: Int, height: Int)
    external fun renderLoop()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        surfaceView = findViewById(R.id.surface_view)
        surfaceView.holder.addCallback(this)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        // Surface is created, init Vulkan
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        if (width == 0 || height == 0) return
        
        try {
            initVulkan(holder.surface, assets, width, height)
            Thread {
                renderLoop()
            }.start()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        destroyVulkan()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        destroyVulkan()
    }
}