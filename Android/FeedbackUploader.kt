package com.example.ecovision

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

/**
 * Sends captured frames to the review backend (Cloudflare Worker → Telegram).
 * Used only when the user has opted in (EcoPreferences.shareFrames).
 */
object FeedbackUploader {
    // Берутся из local.properties (review.apiUrl / review.appKey), см. build.gradle.kts.
    // Ключ из APK можно достать — это защита от случайного мусора, а не настоящая авторизация.
    private val BASE_URL = BuildConfig.REVIEW_API_URL.trimEnd('/')
    private val APP_KEY = BuildConfig.REVIEW_APP_KEY

    /** false, если адрес не прописан в local.properties — тогда ничего не отправляем. */
    val isConfigured: Boolean get() = BASE_URL.isNotEmpty() && APP_KEY.isNotEmpty()

    private const val TAG = "FeedbackUploader"
    private const val MAX_SIDE = 1024
    private const val JPEG_QUALITY = 85
    private const val TIMEOUT_MS = 15_000

    /** @return id записи на сервере или null при ошибке. */
    suspend fun submit(
        context: Context,
        frame: Bitmap,
        prediction: Prediction,
        deviceId: String,
        userLabel: String? = null
    ): Long? = withContext(Dispatchers.IO) {
        runCatching {
            val jpeg = encodeJpeg(frame)
            val boundary = "----EcoVision" + UUID.randomUUID().toString().replace("-", "")
            val conn = open("/v1/submissions").apply {
                setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            }
            DataOutputStream(conn.outputStream).use { out ->
                fun field(name: String, value: String) {
                    out.writeBytes("--$boundary\r\n")
                    out.writeBytes("Content-Disposition: form-data; name=\"$name\"\r\n\r\n")
                    out.write(value.toByteArray(Charsets.UTF_8))
                    out.writeBytes("\r\n")
                }
                field("device_id", deviceId)
                field("predicted_label", prediction.label)
                field("confidence", prediction.confidence.toString())
                field("app_version", appVersion(context))
                if (userLabel != null) field("user_label", userLabel)
                out.writeBytes("--$boundary\r\n")
                out.writeBytes("Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n")
                out.writeBytes("Content-Type: image/jpeg\r\n\r\n")
                out.write(jpeg)
                out.writeBytes("\r\n--$boundary--\r\n")
            }
            val (code, body) = read(conn)
            if (code != 201) error("HTTP $code: $body")
            JSONObject(body).getLong("id")
        }.onFailure { Log.w(TAG, "submit failed", it) }.getOrNull()
    }

    /** Пользователь не согласен с моделью и выбрал другой класс. */
    suspend fun dispute(id: Long, deviceId: String, userLabel: String): Boolean = withContext(Dispatchers.IO) {
        runCatching {
            val body = JSONObject().put("device_id", deviceId).put("user_label", userLabel)
            val (code, text) = postJson("/v1/submissions/$id/dispute", body)
            if (code != 200) error("HTTP $code: $text")
            true
        }.onFailure { Log.w(TAG, "dispute failed", it) }.getOrDefault(false)
    }

    /** Удаляет все кадры этого устройства на сервере. @return сколько удалено или null при ошибке. */
    suspend fun deleteAll(deviceId: String): Int? = withContext(Dispatchers.IO) {
        runCatching {
            val (code, text) = postJson("/v1/delete", JSONObject().put("device_id", deviceId))
            if (code != 200) error("HTTP $code: $text")
            JSONObject(text).getInt("deleted")
        }.onFailure { Log.w(TAG, "delete failed", it) }.getOrNull()
    }

    private fun postJson(path: String, body: JSONObject): Pair<Int, String> {
        val conn = open(path).apply { setRequestProperty("Content-Type", "application/json") }
        conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
        return read(conn)
    }

    private fun open(path: String): HttpURLConnection {
        check(isConfigured) { "Review API is not configured (local.properties)" }
        return (URL(BASE_URL + path).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = TIMEOUT_MS
            readTimeout = TIMEOUT_MS
            setRequestProperty("X-App-Key", APP_KEY)
        }
    }

    private fun read(conn: HttpURLConnection): Pair<Int, String> = try {
        val code = conn.responseCode
        val stream = if (code in 200..299) conn.inputStream else conn.errorStream
        code to (stream?.bufferedReader()?.use { it.readText() } ?: "")
    } finally {
        conn.disconnect()
    }

    /** Уменьшает кадр до 1024 px по длинной стороне: модели всё равно нужно 224×224. */
    private fun encodeJpeg(frame: Bitmap): ByteArray {
        val scale = MAX_SIDE.toFloat() / maxOf(frame.width, frame.height)
        val scaled = if (scale < 1f)
            Bitmap.createScaledBitmap(frame, (frame.width * scale).toInt(), (frame.height * scale).toInt(), true)
        else frame
        return try {
            ByteArrayOutputStream().use { out ->
                scaled.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
                out.toByteArray()
            }
        } finally {
            if (scaled !== frame) scaled.recycle()
        }
    }

    private fun appVersion(context: Context): String =
        runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }
            .getOrNull() ?: "?"
}
