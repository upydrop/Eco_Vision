package com.example.ecovision

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch

@Composable
fun EcoPageHeader(title: String, p: EcoPalette, onBack: () -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack, "Back", tint = p.text) }
        Spacer(Modifier.width(8.dp))
        Text(title, fontSize = 25.sp, fontWeight = FontWeight.Bold, color = p.text)
    }
}

@Composable
private fun EcoSwitchRow(title: String, description: String, checked: Boolean, p: EcoPalette, onCheckedChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, color = p.text, fontWeight = FontWeight.SemiBold)
            Text(description, color = p.muted, fontSize = 12.sp)
        }
        Switch(checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun EcoSection(title: String, p: EcoPalette, content: @Composable ColumnScope.() -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, color = p.text, fontSize = 18.sp, fontWeight = FontWeight.Bold)
        Column(Modifier.fillMaxWidth().background(p.panel, RoundedCornerShape(24.dp)).padding(18.dp), content = content)
    }
}

@Composable
fun SettingsScreen(
    p: EcoPalette,
    prefs: EcoPreferences,
    theme: String,
    onThemeChange: (String) -> Unit,
    onBack: () -> Unit,
    onDeveloper: () -> Unit
) {
    var stabilize by remember { mutableStateOf(prefs.stabilize) }
    var animations by remember { mutableStateOf(prefs.animations) }
    var unlocked by remember { mutableStateOf(prefs.developerUnlocked) }
    var shareFrames by remember { mutableStateOf(prefs.shareFrames) }
    var showConsent by remember { mutableStateOf(false) }
    var deleteStatus by remember { mutableStateOf<String?>(null) }
    var deleting by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    Column(Modifier.fillMaxSize().background(p.background).statusBarsPadding()
        .verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)) {
        EcoPageHeader("Settings", p, onBack)
        EcoSection("Appearance", p) {
            Text("Theme", color = p.text, fontWeight = FontWeight.SemiBold)
            listOf("system" to "System default", "light" to "Light", "dark" to "Dark").forEach { (value, title) ->
                Row(Modifier.fillMaxWidth().clickable { onThemeChange(value) }.padding(vertical = 5.dp), verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = theme == value, onClick = { onThemeChange(value) })
                    Text(title, color = p.text)
                }
            }
            HorizontalDivider(color = p.soft)
            EcoSwitchRow("Smooth animations", "Interface and result animations", animations, p) {
                animations = it; prefs.animations = it
            }
        }
        EcoSection("Camera & scanning", p) {
            EcoSwitchRow("Stable result", "Reduces jumping between categories", stabilize, p) {
                stabilize = it; prefs.stabilize = it
            }
            Text(if (shareFrames) "Predictions run on the device. Captured frames are shared for review."
                else "Predictions run entirely on the device. Frames are not saved.", color = p.muted, fontSize = 12.sp)
        }
        EcoSection("Help improve EcoVision", p) {
            EcoSwitchRow("Share captured frames",
                "Send captured frames and your corrections to the team to check and improve the model",
                shareFrames, p) {
                if (it) showConsent = true
                else { shareFrames = false; prefs.shareFrames = false; prefs.shareConsentAsked = true }
            }
            HorizontalDivider(color = p.soft)
            Row(Modifier.fillMaxWidth().clickable(enabled = !deleting) {
                deleting = true
                deleteStatus = "Deleting…"
                scope.launch {
                    val n = FeedbackUploader.deleteAll(prefs.deviceId)
                    deleteStatus = when (n) {
                        null -> "Could not delete. Check your connection and try again"
                        0 -> "No shared frames found"
                        else -> "Deleted $n shared frame(s)"
                    }
                    deleting = false
                }
            }.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Delete my shared frames", color = p.text, fontWeight = FontWeight.SemiBold)
                    Text(deleteStatus ?: "Removes every frame sent from this device", color = p.muted, fontSize = 12.sp)
                }
                Icon(Icons.Rounded.ChevronRight, null, tint = p.text)
            }
        }
        if (unlocked) {
            EcoSection("Advanced", p) {
                Row(Modifier.fillMaxWidth().clickable(onClick = onDeveloper).padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("For developers", color = p.text, fontWeight = FontWeight.SemiBold)
                        Text("Percentages and prediction debugging", color = p.muted, fontSize = 12.sp)
                    }
                    Icon(Icons.Rounded.ChevronRight, null, tint = p.text)
                }
            }
        }
        Spacer(Modifier.height(12.dp))
    }
    if (showConsent) ShareConsentDialog(
        onAllow = { shareFrames = true; prefs.shareFrames = true; prefs.shareConsentAsked = true; showConsent = false },
        onDeny = { shareFrames = false; prefs.shareFrames = false; prefs.shareConsentAsked = true; showConsent = false }
    )
}

@Composable
fun DeveloperScreen(p: EcoPalette, prefs: EcoPreferences, onBack: () -> Unit) {
    var showPercent by remember { mutableStateOf(prefs.showPercent) }
    var showRaw by remember { mutableStateOf(prefs.showRaw) }
    var custom by remember { mutableStateOf(prefs.customThresholds) }
    var high by remember { mutableFloatStateOf(prefs.highThreshold) }
    var low by remember { mutableFloatStateOf(prefs.lowThreshold) }
    Column(Modifier.fillMaxSize().background(p.background).statusBarsPadding()
        .verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
        EcoPageHeader("Developer mode", p, onBack)
        EcoSection("Display", p) {
            EcoSwitchRow("Show percentages", "Exact model confidence score", showPercent, p) {
                showPercent = it; prefs.showPercent = it
            }
            HorizontalDivider(color = p.soft)
            EcoSwitchRow("Raw predictions", "Raw value of the top prediction", showRaw, p) {
                showRaw = it; prefs.showRaw = it
            }
        }
        EcoSection("Color thresholds", p) {
            EcoSwitchRow("Custom thresholds", "Change status boundaries", custom, p) {
                custom = it; prefs.customThresholds = it
            }
            if (custom) {
                Text("Green: from ${(high * 100).toInt()}%", color = p.text)
                // В SharedPreferences пишем, когда палец отпущен, а не на каждый шаг ползунка.
                Slider(value = high, onValueChange = { high = it.coerceAtLeast(low + 0.05f) },
                    onValueChangeFinished = { prefs.highThreshold = high }, valueRange = 0.55f..1.0f)
                Text("Yellow: from ${(low * 100).toInt()}%", color = p.text)
                Slider(value = low, onValueChange = { low = it.coerceAtMost(high - 0.05f) },
                    onValueChangeFinished = { prefs.lowThreshold = low }, valueRange = 0.05f..0.90f)
            } else Text("Default: green from 80%, yellow from 50%.", color = p.muted, fontSize = 13.sp)
        }
        Text("The confidence score is the model's output, not a guarantee of correct classification.", color = p.muted, fontSize = 12.sp)
    }
}

@Composable
fun AboutScreen(p: EcoPalette, prefs: EcoPreferences, onBack: () -> Unit, onUnlock: () -> Unit) {
    var taps by remember { mutableIntStateOf(0) }
    Column(Modifier.fillMaxSize().background(p.background).statusBarsPadding().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)) {
        EcoPageHeader("About", p, onBack)
        Column(Modifier.fillMaxWidth().background(p.panel, RoundedCornerShape(28.dp)).padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text("EcoVision", color = p.accent, fontSize = 35.sp, fontWeight = FontWeight.Black)
            Text("SORT SMARTER", color = p.muted, letterSpacing = 3.sp)
            Text("Material recognition with a neural network, right on your device.", color = p.text)
            Text(if (prefs.shareFrames) "Captured frames are shared with the team for review (you can turn this off in Settings)."
                else "The camera analyzes frames without saving photos.", color = p.muted, fontSize = 13.sp)
            Text("Created by 'MusorDrop':\nLead engineers: Valery Semyannikov, Alexander Babarik\nCollaborator: ChatGPT", color = p.muted, fontSize = 13.sp)
            Text("Version 1.2.1", color = p.text, modifier = Modifier.clickable {
                if (!prefs.developerUnlocked) {
                    taps++
                    if (taps >= 7) { prefs.developerUnlocked = true; onUnlock() }
                }
            }.padding(12.dp))
            if (prefs.developerUnlocked) Text("Developer mode is unlocked in Settings", color = p.accent, fontSize = 12.sp)
            else if (taps >= 3) Text("Taps until developer mode: ${7 - taps}", color = p.muted, fontSize = 12.sp)
        }
    }
}