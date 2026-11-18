package com.example.ecovision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

data class Prediction(
    val label: String,
    val confidence: Float
)

/**
 * Не потокобезопасен: [classify] вызывается только с одного потока (executor камеры),
 * поэтому буферы ввода/вывода выделяются один раз и переиспользуются для каждого кадра.
 */
class WasteClassifier(context: Context) : Closeable {

    private val labels: List<String> =
        context.assets.open("labels.json")
            .bufferedReader()
            .use { reader ->
                org.json.JSONArray(reader.readText()).let { array ->
                    List(array.length()) { index ->
                        array.getString(index)
                    }
                }
            }

    private val interpreter: Interpreter

    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val floats = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)
    private val input: ByteBuffer = ByteBuffer.allocateDirect(floats.size * 4).order(ByteOrder.nativeOrder())
    private val inputFloats: FloatBuffer = input.asFloatBuffer()
    private val output: Array<FloatArray>
    private val matrix = Matrix()

    init {
        val modelBuffer = context.assets
            .openFd("waste_classifier.tflite")
            .use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                    channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        descriptor.startOffset,
                        descriptor.declaredLength
                    )
                }
            }

        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }

        interpreter = Interpreter(modelBuffer, options)

        require(interpreter.getInputTensor(0).shape()
            .contentEquals(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3))) {
            "Unexpected model input shape"
        }

        require(labels.size == interpreter.getOutputTensor(0).shape().last()) {
            "Number of labels does not match model output"
        }

        output = Array(1) { FloatArray(labels.size) }
    }

    /**
     * @param rotationDegrees поворот кадра камеры. Масштабирование до 224×224 и поворот делаются
     * одной операцией: результат тот же, что «повернуть полный кадр, потом сжать», но поворачивается
     * уже маленькая картинка, а не весь кадр.
     */
    fun classify(bitmap: Bitmap, rotationDegrees: Int = 0): Prediction {
        matrix.reset()
        matrix.postScale(INPUT_SIZE.toFloat() / bitmap.width, INPUT_SIZE.toFloat() / bitmap.height)
        if (rotationDegrees != 0) matrix.postRotate(rotationDegrees.toFloat())

        var resized = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (resized.width != INPUT_SIZE || resized.height != INPUT_SIZE) {
            // Страховка от округления на нестандартных углах.
            val exact = Bitmap.createScaledBitmap(resized, INPUT_SIZE, INPUT_SIZE, true)
            if (exact !== resized && resized !== bitmap) resized.recycle()
            resized = exact
        }

        try {
            resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        } finally {
            if (resized !== bitmap) resized.recycle()
        }

        var i = 0
        for (pixel in pixels) {
            floats[i++] = ((pixel shr 16) and 0xFF).toFloat()
            floats[i++] = ((pixel shr 8) and 0xFF).toFloat()
            floats[i++] = (pixel and 0xFF).toFloat()
        }
        inputFloats.rewind()
        inputFloats.put(floats)
        input.rewind()

        interpreter.run(input, output)

        val scores = output[0]
        var bestIndex = 0
        for (index in 1 until scores.size) if (scores[index] > scores[bestIndex]) bestIndex = index

        return Prediction(
            label = labels[bestIndex],
            confidence = scores[bestIndex].coerceIn(0f, 1f)
        )
    }

    override fun close() {
        interpreter.close()
    }

    private companion object {
        const val INPUT_SIZE = 224
    }
}