package com.example.ecovision

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.Replay
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.launch

private enum class Screen { Menu, Camera, Settings, About, Developer, Map  }

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefs = EcoPreferences(this)
        setContent {
            var screen by remember { mutableStateOf(Screen.Menu) }
            var themeMode by remember { mutableStateOf(prefs.theme) }
            var settingsVersion by remember { mutableIntStateOf(0) }
            val dark = isDarkEco(themeMode)
            val palette = if (dark) DarkEco else LightEco
            EcoVisionTheme(dark) {
                val back = { screen = if (screen == Screen.Developer) Screen.Settings else Screen.Menu }
                BackHandler(screen != Screen.Menu) { back() }
                // Refresh user-configurable flags when returning from settings.
                @Suppress("UNUSED_VARIABLE") val v = settingsVersion
                when (screen) {
                    Screen.Menu -> MainMenuScreen(palette, prefs.animations,
                        onCamera = { screen = Screen.Camera },
                        onSettings = { screen = Screen.Settings },
                        onAbout = { screen = Screen.About },
                        onMap = { screen = Screen.Map })
                    Screen.Map -> RecyclingMapScreen(palette, onBack = { screen = Screen.Menu })
                    Screen.Camera -> WasteCameraScreen(palette, prefs, onBack = { screen = Screen.Menu }, onSettings = { screen = Screen.Settings })
                    Screen.Settings -> SettingsScreen(palette, prefs, themeMode,
                        onThemeChange = { themeMode = it; prefs.theme = it },
                        onBack = { settingsVersion++; screen = Screen.Menu },
                        onDeveloper = { screen = Screen.Developer })
                    Screen.About -> AboutScreen(palette, prefs, onBack = { screen = Screen.Menu },
                        onUnlock = { settingsVersion++ })
                    Screen.Developer -> DeveloperScreen(palette, prefs, onBack = { screen = Screen.Settings })
                }
            }
        }
    }
}

/** Русские названия классов модели (labels.json). */
private val LABEL_TITLES = mapOf(
    "glass" to "Glass",
    "metal" to "Metal",
    "paper" to "Paper",
    "plastic" to "Plastic",
    "trash" to "Other waste"
)

private fun labelTitle(label: String): String =
    LABEL_TITLES[label] ?: label.replaceFirstChar { it.titlecase(Locale.getDefault()) }

/** Снимок настроек камеры на время жизни экрана. */
private data class CameraSettings(
    val low: Float,
    val high: Float,
    val stabilize: Boolean,
    val animations: Boolean,
    val showPercent: Boolean,
    val showRaw: Boolean
) {
    companion object {
        fun from(prefs: EcoPreferences) = CameraSettings(
            low = if (prefs.customThresholds) prefs.lowThreshold else 0.5f,
            high = if (prefs.customThresholds) prefs.highThreshold else 0.8f,
            stabilize = prefs.stabilize,
            animations = prefs.animations,
            showPercent = prefs.developerUnlocked && prefs.showPercent,
            showRaw = prefs.developerUnlocked && prefs.showRaw
        )
    }
}

@Composable
fun WasteCameraScreen(
    p: EcoPalette,
    prefs: EcoPreferences,
    onBack: () -> Unit,
    onSettings: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var hasPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it }
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }
    val executor = remember { Executors.newSingleThreadExecutor() }
    // Модель грузится на потоке камеры при первом кадре, а не на UI-потоке при открытии экрана.
    // Доступ к ней только с этого executor'а, поэтому синхронизация не нужна.
    val classifier = remember { lazy(LazyThreadSafetyMode.NONE) { WasteClassifier(context.applicationContext) } }
    val cameraActive = remember { AtomicBoolean(true) }
    // 0 = live, 1 = capture next frame, 2 = frozen frame
    val captureState = remember { AtomicInteger(0) }
    var frozenBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var freezePending by remember { mutableStateOf(false) }
    var prediction by remember { mutableStateOf<Prediction?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    // Require a few consecutive equivalent predictions to reduce flicker.
    var candidate by remember { mutableStateOf<String?>(null) }
    var candidateCount by remember { mutableIntStateOf(0) }
    var accepted by remember { mutableStateOf<Prediction?>(null) }
    // Настройки нельзя поменять, не уйдя с экрана камеры (он при этом пересоздаётся),
    // поэтому читаем SharedPreferences один раз, а не на каждую перерисовку (~каждый кадр).
    val settings = remember { CameraSettings.from(prefs) }
    val low = settings.low
    val high = settings.high

    // ── Отправка кадров на проверку (только с согласия пользователя) ──
    val scope = rememberCoroutineScope()
    var shareFrames by remember { mutableStateOf(prefs.shareFrames) }
    var showConsent by remember { mutableStateOf(!prefs.shareConsentAsked) }
    var showDispute by remember { mutableStateOf(false) }
    var pendingDisputeLabel by remember { mutableStateOf<String?>(null) }
    var capturedPrediction by remember { mutableStateOf<Prediction?>(null) }
    var submission by remember { mutableStateOf<Deferred<Long?>?>(null) }
    var feedbackStatus by remember { mutableStateOf<String?>(null) }
    // Номер текущего зафиксированного кадра: ответы по старому кадру не должны менять экран нового.
    var captureGen by remember { mutableIntStateOf(0) }

    fun startUpload(frame: Bitmap, result: Prediction) {
        val gen = captureGen
        val job = scope.async { FeedbackUploader.submit(context, frame, result, prefs.deviceId) }
        submission = job
        feedbackStatus = "Sending frame for review…"
        scope.launch {
            val id = job.await()
            if (gen == captureGen) feedbackStatus =
                if (id != null) "Frame sent for review" else "Could not send the frame"
        }
    }

    fun sendCorrection(userLabel: String) {
        val frame = frozenBitmap ?: return
        val result = capturedPrediction ?: return
        val gen = captureGen
        val job = submission
        feedbackStatus = "Sending your correction…"
        scope.launch {
            val existingId = job?.await()
            val ok = if (existingId != null) FeedbackUploader.dispute(existingId, prefs.deviceId, userLabel)
            else {
                val id = FeedbackUploader.submit(context, frame, result, prefs.deviceId, userLabel)
                if (gen == captureGen && id != null) submission = scope.async { id }
                id != null
            }
            if (gen == captureGen) feedbackStatus =
                if (ok) "Thanks! Your correction was sent for review" else "Could not send. Check your connection"
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraActive.set(false)
            captureState.set(2)
            // Stop accepting frames before disposing the interpreter.
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener({
                runCatching { future.get().unbindAll() }
            }, ContextCompat.getMainExecutor(context))
            // The close operation must run *after* any queued inference on the SAME executor.
            executor.execute { if (classifier.isInitialized()) runCatching { classifier.value.close() } }
            executor.shutdown()
        }
    }
    LaunchedEffect(hasPermission) {
        if (hasPermission && cameraActive.get()) startCamera(previewView, lifecycleOwner, executor, classifier,
            cameraActive, captureState,
            onPrediction = { value ->
                if (cameraActive.get() && captureState.get() == 0) {
                    prediction = value
                    errorMessage = null
                    if (!settings.stabilize) {
                        accepted = value
                    } else {
                        val status = if (value.confidence < low) "thinking" else value.label
                        if (candidate == status) candidateCount++ else { candidate = status; candidateCount = 1 }
                        if (candidateCount >= 3) accepted = value
                        else if (accepted?.label == value.label && value.confidence >= low) accepted = value
                    }
                }
            },
            onCaptured = { frame, result ->
                if (cameraActive.get() && captureState.get() == 2) {
                    frozenBitmap = frame
                    freezePending = false
                    prediction = result
                    accepted = result
                    candidate = result.label
                    candidateCount = 3
                    errorMessage = null
                    capturedPrediction = result
                    if (shareFrames) startUpload(frame, result)
                }
            },
            onError = { if (cameraActive.get()) { errorMessage = it; freezePending = false } })
    }
    val displayed = if (settings.stabilize) accepted else prediction
    val confidence = displayed?.confidence ?: 0f
    val thinking = displayed == null || confidence < low
    val statusColor = when {
        thinking -> p.muted
        confidence >= high -> p.confident
        else -> p.uncertain
    }
    val animatedColor by animateColorAsState(statusColor, tween(if (settings.animations) 300 else 0), label = "resultColor")
    val progress by animateFloatAsState(confidence, tween(if (settings.animations) 300 else 0), label = "confidence")

    Column(Modifier.fillMaxSize().background(p.background).statusBarsPadding().navigationBarsPadding()
        .padding(horizontal = 16.dp, vertical = 10.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            IconButton(onClick = onBack, modifier = Modifier.background(p.panel, RoundedCornerShape(50))) {
                Icon(Icons.AutoMirrored.Rounded.ArrowBack, "Back", tint = p.text)
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("EcoVision", color = p.text, fontSize = 25.sp, fontWeight = FontWeight.Black)
                Text("SMART SCANNER", color = p.accent, fontSize = 10.sp, letterSpacing = 2.sp)
            }
            IconButton(onClick = onSettings, modifier = Modifier.background(p.panel, RoundedCornerShape(50))) {
                Icon(Icons.Rounded.Settings, "Settings", tint = p.text)
            }
        }
        Spacer(Modifier.height(16.dp))
        if (!hasPermission) {
            Spacer(Modifier.weight(1f))
            Text("Camera access is required for recognition", color = p.text)
            Spacer(Modifier.height(12.dp))
            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }
            Spacer(Modifier.weight(1f))
        } else {
            Box(modifier = Modifier.fillMaxWidth().weight(1f)
                .clip(RoundedCornerShape(26.dp))) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                frozenBitmap?.let { frame ->
                    Image(
                        bitmap = frame.asImageBitmap(),
                        contentDescription = "Captured frame",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
            Spacer(Modifier.height(12.dp))
            Button(
                enabled = !freezePending,
                onClick = {
                    if (captureState.get() == 2) {
                        captureState.set(0)
                        captureGen++
                        capturedPrediction = null
                        submission = null
                        feedbackStatus = null
                        showDispute = false
                        frozenBitmap = null
                        prediction = null
                        accepted = null
                        candidate = null
                        candidateCount = 0
                        errorMessage = null
                    } else if (captureState.compareAndSet(0, 1)) {
                        freezePending = true
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = p.accent),
                shape = RoundedCornerShape(18.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    if (frozenBitmap != null) Icons.Rounded.Replay else Icons.Rounded.CameraAlt,
                    contentDescription = null
                )
                Spacer(Modifier.width(8.dp))
                Text(when {
                    frozenBitmap != null -> "Resume scanning"
                    freezePending -> "Capturing frame…"
                    else -> "Capture frame"
                })
            }
            Spacer(Modifier.height(12.dp))
            Column(Modifier.fillMaxWidth().background(p.panel, RoundedCornerShape(24.dp)).padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("DETECTED", color = p.muted, fontSize = 11.sp, letterSpacing = 2.sp)
                // Фиксированная высота строки: панель не прыгает при смене «думает…» ↔ результат.
                Row(Modifier.fillMaxWidth().height(36.dp), verticalAlignment = Alignment.CenterVertically) {
                    val error = errorMessage
                    // Ошибка показывается в строке результата, чтобы высота панели не менялась.
                    if (error != null) Text(error, color = MaterialTheme.colorScheme.error, fontSize = 14.sp,
                        maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                    else Text(if (thinking) "Model is thinking…" else labelTitle(displayed!!.label),
                        color = animatedColor, fontWeight = FontWeight.Bold,
                        fontSize = if (thinking) 20.sp else 26.sp, maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                    if (error == null && !thinking) Icon(Icons.Rounded.CheckCircle, null, tint = animatedColor)
                }
                LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().height(8.dp)
                    .clip(RoundedCornerShape(50)), color = animatedColor, trackColor = p.soft)
                // Строки разработчика видны всегда, когда включены (с «—» до первого результата).
                if (settings.showPercent)
                    Text("Model confidence: " + (displayed?.let { String.format(Locale.US, "%.1f%%", confidence * 100) } ?: "—"),
                        color = p.muted, fontSize = 13.sp)
                if (settings.showRaw)
                    Text("Raw score: " + (displayed?.let { String.format(Locale.US, "%.6f", confidence) } ?: "—"),
                        color = p.muted, fontSize = 12.sp)
            }
            if (frozenBitmap != null && capturedPrediction != null) {
                Row(Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(feedbackStatus ?: "", color = p.muted, fontSize = 12.sp, maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                    TextButton(onClick = { showDispute = true }) { Text("Wrong result?") }
                }
            } else Spacer(Modifier.height(10.dp))
            Text(if (shareFrames) "On-device analysis · Captured frames are shared for review"
                else "On-device analysis · Frame kept in memory only", color = p.muted, fontSize = 11.sp)
        }
    }

    if (showConsent) ShareConsentDialog(
        onAllow = {
            prefs.shareFrames = true; prefs.shareConsentAsked = true
            shareFrames = true; showConsent = false
            pendingDisputeLabel?.let { sendCorrection(it) }
            pendingDisputeLabel = null
        },
        onDeny = {
            prefs.shareFrames = false; prefs.shareConsentAsked = true
            shareFrames = false; showConsent = false
            if (pendingDisputeLabel != null) feedbackStatus = "Not sent: frame sharing is off"
            pendingDisputeLabel = null
        }
    )

    if (showDispute) capturedPrediction?.let { result ->
        AlertDialog(
            onDismissRequest = { showDispute = false },
            title = { Text("What is it actually?") },
            text = {
                Column {
                    Text("Model said: ${labelTitle(result.label)}. Choose the correct category:",
                        color = p.muted, fontSize = 13.sp)
                    Spacer(Modifier.height(8.dp))
                    LABEL_TITLES.keys.filter { it != result.label }.forEach { label ->
                        TextButton(onClick = {
                            showDispute = false
                            if (shareFrames) sendCorrection(label)
                            else { pendingDisputeLabel = label; showConsent = true }
                        }, modifier = Modifier.fillMaxWidth()) { Text(labelTitle(label)) }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { showDispute = false }) { Text("Cancel") } }
        )
    }
}

/** Согласие на отправку кадров. Тексты интерфейса — только на английском. */
@Composable
fun ShareConsentDialog(onAllow: () -> Unit, onDeny: () -> Unit) {
    AlertDialog(
        onDismissRequest = {}, // явный выбор обязателен
        title = { Text("Help improve EcoVision") },
        text = {
            Text("When you capture a frame, it can be sent to the EcoVision team together with the model's " +
                "answer and any correction you make. We check whether recognition was right and use the " +
                "frames to improve the model.\n\nFrames are linked only to a random ID of this install, not " +
                "to your name. Please avoid capturing people or personal information. You can turn this off " +
                "and delete your shared frames at any time in Settings.")
        },
        confirmButton = { TextButton(onClick = onAllow) { Text("Allow") } },
        dismissButton = { TextButton(onClick = onDeny) { Text("No thanks") } }
    )
}

/** ~8 кадров/с для живого распознавания. */
private const val MIN_INFERENCE_INTERVAL_MS = 125L

private fun startCamera(
    previewView: PreviewView,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    executor: ExecutorService,
    classifier: Lazy<WasteClassifier>,
    cameraActive: AtomicBoolean,
    captureState: AtomicInteger,
    onCaptured: (Bitmap, Prediction) -> Unit,
    onPrediction: (Prediction) -> Unit,
    onError: (String) -> Unit
) {
    val context = previewView.context
    val providerFuture = ProcessCameraProvider.getInstance(context)

    providerFuture.addListener({
        try {
            val provider = providerFuture.get()
            if (!cameraActive.get()) return@addListener

            val preview = Preview.Builder()
                .build()
                .also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(
                    ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                )
                // CameraX сам конвертирует YUV→RGBA нативно, toBitmap() становится простым копированием.
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            // Обращение только с потока executor'а — синхронизация не нужна.
            var lastInferenceAt = 0L
            analysis.setAnalyzer(executor) { imageProxy ->
                if (!cameraActive.get() || captureState.get() == 2) {
                    imageProxy.close()
                    return@setAnalyzer
                }
                // Живое распознавание не чаще ~8 раз в секунду: меньше нагрев и расход батареи.
                // Запрос на фиксацию кадра (state 1) обрабатывается сразу, без ожидания.
                val now = android.os.SystemClock.elapsedRealtime()
                if (captureState.get() == 0 && now - lastInferenceAt < MIN_INFERENCE_INTERVAL_MS) {
                    imageProxy.close()
                    return@setAnalyzer
                }
                lastInferenceAt = now
                var bitmap: Bitmap? = null

                try {
                    bitmap = imageProxy.toBitmap()
                    val rotation = imageProxy.imageInfo.rotationDegrees

                    val isCapturing = captureState.get() == 1
                    // Полноразмерный повёрнутый кадр нужен только для фиксации — для модели
                    // поворачивается уже уменьшенная до 224×224 картинка (внутри classify).
                    val frozenFrame = if (isCapturing) rotateCopy(bitmap, rotation) else null
                    val result = try {
                        classifier.value.classify(bitmap, rotation)
                    } catch (e: Exception) {
                        frozenFrame?.recycle()
                        throw e
                    }

                    if (isCapturing && captureState.compareAndSet(1, 2)) {
                        previewView.post {
                            if (cameraActive.get() && captureState.get() == 2) onCaptured(frozenFrame!!, result)
                            else frozenFrame?.recycle()
                        }
                    } else {
                        frozenFrame?.recycle()
                        previewView.post {
                            if (cameraActive.get() && captureState.get() == 0) onPrediction(result)
                        }
                    }
                } catch (e: Exception) {
                    captureState.compareAndSet(1, 0)
                    previewView.post {
                        if (cameraActive.get()) onError(
                            e.message ?: "Could not process the frame"
                        )
                    }
                } finally {
                    bitmap?.recycle()
                    imageProxy.close()
                }
            }

            if (!cameraActive.get()) return@addListener
            provider.unbindAll()

            if (!cameraActive.get()) return@addListener
            provider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        } catch (e: Exception) {
            if (cameraActive.get()) onError(
                e.message ?: "Could not start the camera"
            )
        }
    }, ContextCompat.getMainExecutor(context))
}

/** Независимая (не переиспользуемая) копия кадра с учётом поворота — для зафиксированного снимка. */
private fun rotateCopy(source: Bitmap, rotation: Int): Bitmap =
    if (rotation == 0) source.copy(Bitmap.Config.ARGB_8888, false)
    else Bitmap.createBitmap(
        source, 0, 0, source.width, source.height,
        Matrix().apply { postRotate(rotation.toFloat()) }, true
    )