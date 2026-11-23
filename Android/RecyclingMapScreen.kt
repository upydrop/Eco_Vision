package com.example.ecovision

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.RectF
import android.location.Location
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.os.CancellationSignal
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.MyLocation
import androidx.compose.material.icons.rounded.Navigation
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.maplibre.android.MapLibre
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.expressions.Expression
import org.maplibre.android.style.layers.CircleLayer
import org.maplibre.android.style.layers.PropertyFactory
import org.maplibre.android.style.layers.SymbolLayer
import org.maplibre.android.style.sources.GeoJsonOptions
import org.maplibre.android.style.sources.GeoJsonSource
import org.maplibre.geojson.Feature
import org.maplibre.geojson.FeatureCollection
import org.maplibre.geojson.Point
import java.util.Locale

private const val MAP_SOURCE = "recycling-source"
private const val LAYER_CLUSTERS = "recycling-clusters"
private const val LAYER_CLUSTER_COUNT = "recycling-cluster-count"
private const val LAYER_SINGLE = "recycling-single"
private const val LAYER_SELECTED = "recycling-selected"
private const val MAP_STYLE = "https://tiles.openfreemap.org/styles/liberty"
// Стиль OpenFreeMap отдаёт только шрифты Noto Sans; без явного шрифта подписи кластеров не рисуются.
private val MAP_FONT = arrayOf("Noto Sans Bold")

private const val COLOR_POINT = "#2E9E5B"   // контейнеры
private const val COLOR_DEPOSIT = "#2F7FD8" // тараматы / приём тары
private const val COLOR_SITE = "#E08A1E"    // площадки (в т.ч. «другое»)

/** Вид пункта: обычные контейнеры, приём депозитной тары или площадка сортированных отходов. */
private enum class PointKind(val key: String, val title: String) {
    Containers("point", "Sorting containers"),
    Deposit("deposit", "Deposit return point"),
    Site("site", "Sorted waste site")
}

private data class RecyclingPoint(
    val id: String,
    val kind: PointKind,
    val lat: Double,
    val lon: Double,
    val address: String,
    val operator: String,
    /** Только категории, которые распознаёт модель: paper, plastic, glass, metal. */
    val materials: Set<String>,
    /** Прочие принимаемые отходы (переведены на русский). */
    val extras: List<String>,
    val store: String?,
    val note: String?,
    val hours: List<Pair<String, String>>
)

/** Фильтры = классы модели (labels.json: glass, metal, paper, plastic, trash) + тарамат. */
private val filters = listOf(
    "all" to "All",
    "paper" to "Paper",
    "plastic" to "Plastic",
    "glass" to "Glass",
    "metal" to "Metal",
    "trash" to "Other",
    "deposit" to "Deposit"
)

private val MAIN_MATERIALS = linkedMapOf(
    "paper" to "Paper", "plastic" to "Plastic", "glass" to "Glass", "metal" to "Metal"
)

private val EXTRA_MATERIALS = linkedMapOf(
    "bulky" to "bulky waste", "electronics" to "electronics", "batteries" to "batteries",
    "lamps" to "lamps", "hazardous" to "hazardous waste", "paint" to "paint", "tires" to "tires",
    "oil" to "oil", "construction" to "construction waste", "wood" to "wood",
    "textiles" to "textiles", "bio" to "bio waste", "cooking_oil" to "cooking oil",
    "asbestos" to "asbestos"
)

private fun RecyclingPoint.matches(filter: String): Boolean = when (filter) {
    "all" -> true
    "deposit" -> kind == PointKind.Deposit
    // trash = всё, что не бумага/пластик/стекло/металл: площадки и контейнеры для прочих отходов.
    "trash" -> kind == PointKind.Site || (kind == PointKind.Containers && extras.isNotEmpty())
    // Депозитная тара НЕ считается пластиком/стеклом/металлом автоматически.
    else -> filter in materials
}

/**
 * Читает очищенный recycling_points.json. Формат точки:
 * { id, kind: point|deposit|site, lat, lon, address,
 *   materials?: [paper, plastic, glass, metal, bulky, ...], operator?, store?, note?,
 *   hours?: [["Tue–Sun", "10:00–19:00"], ...] }
 * Переводы, разбор примечаний и сжатие часов уже сделаны в самом файле.
 */
private fun loadPoints(context: Context): List<RecyclingPoint> {
    val root = JSONObject(context.assets.open("recycling_points.json").bufferedReader().use { it.readText() })
    val array = root.getJSONArray("points")
    return List(array.length()) { i ->
        val o = array.getJSONObject(i)
        val kind = when (o.getString("kind")) {
            "deposit" -> PointKind.Deposit
            "site" -> PointKind.Site
            else -> PointKind.Containers
        }
        val all = o.optJSONArray("materials")?.let { a -> List(a.length()) { a.getString(it) } }.orEmpty()
        val hours = o.optJSONArray("hours")?.let { a ->
            List(a.length()) { a.getJSONArray(it).let { d -> d.getString(0) to d.getString(1) } }
        }.orEmpty()
        RecyclingPoint(
            id = o.getString("id"),
            kind = kind,
            lat = o.getDouble("lat"),
            lon = o.getDouble("lon"),
            address = o.optString("address"),
            operator = o.optString("operator"),
            materials = all.filter { it in MAIN_MATERIALS }.toSet(),
            extras = EXTRA_MATERIALS.filterKeys { it in all }.values.toList(),
            store = o.optString("store").ifBlank { null },
            note = o.optString("note").ifBlank { null },
            hours = hours
        )
    }
}

/** Точки читаются из assets один раз за запуск приложения; повторное открытие карты — мгновенное. */
private object PointsCache {
    @Volatile var points: List<RecyclingPoint>? = null
}

private fun pointsGeoJson(points: List<RecyclingPoint>): FeatureCollection =
    FeatureCollection.fromFeatures(points.map { p ->
        Feature.fromGeometry(Point.fromLngLat(p.lon, p.lat)).apply {
            addStringProperty("id", p.id)
            addStringProperty("kind", p.kind.key)
        }
    })

/** Источник и слои создаются ОДИН раз при загрузке стиля, дальше меняются только данные. */
private fun setupLayers(style: Style) {
    if (style.getSource(MAP_SOURCE) != null) return
    style.addSource(
        GeoJsonSource(
            MAP_SOURCE, FeatureCollection.fromFeatures(emptyList()),
            GeoJsonOptions().withCluster(true).withClusterRadius(50).withClusterMaxZoom(14)
        )
    )
    style.addLayer(
        CircleLayer(LAYER_CLUSTERS, MAP_SOURCE).withProperties(
            PropertyFactory.circleColor("#377C55"),
            PropertyFactory.circleRadius(
                Expression.step(
                    Expression.get("point_count"), Expression.literal(16f),
                    Expression.stop(20, 20f), Expression.stop(100, 25f), Expression.stop(500, 30f)
                )
            ),
            PropertyFactory.circleOpacity(0.92f),
            PropertyFactory.circleStrokeColor("#E6FFF0"),
            PropertyFactory.circleStrokeWidth(2f)
        ).withFilter(Expression.has("point_count"))
    )
    style.addLayer(
        SymbolLayer(LAYER_CLUSTER_COUNT, MAP_SOURCE).withProperties(
            PropertyFactory.textField(Expression.toString(Expression.get("point_count_abbreviated"))),
            PropertyFactory.textFont(MAP_FONT),
            PropertyFactory.textSize(12f),
            PropertyFactory.textColor("#FFFFFF"),
            PropertyFactory.textIgnorePlacement(true),
            PropertyFactory.textAllowOverlap(true)
        ).withFilter(Expression.has("point_count"))
    )
    style.addLayer(
        CircleLayer(LAYER_SINGLE, MAP_SOURCE).withProperties(
            PropertyFactory.circleColor(
                Expression.match(
                    Expression.get("kind"), Expression.color(android.graphics.Color.parseColor(COLOR_POINT)),
                    Expression.stop("deposit", Expression.color(android.graphics.Color.parseColor(COLOR_DEPOSIT))),
                    Expression.stop("site", Expression.color(android.graphics.Color.parseColor(COLOR_SITE)))
                )
            ),
            PropertyFactory.circleRadius(7.5f),
            PropertyFactory.circleStrokeColor("#FFFFFF"),
            PropertyFactory.circleStrokeWidth(2f)
        ).withFilter(Expression.not(Expression.has("point_count")))
    )
    style.addLayer(
        CircleLayer(LAYER_SELECTED, MAP_SOURCE).withProperties(
            PropertyFactory.circleColor("#193D2A"),
            PropertyFactory.circleOpacity(0f),
            PropertyFactory.circleRadius(13f),
            PropertyFactory.circleStrokeColor("#193D2A"),
            PropertyFactory.circleStrokeWidth(3f)
        ).withFilter(Expression.eq(Expression.get("id"), Expression.literal("")))
    )
}

private fun distanceMeters(location: Location, point: RecyclingPoint): Float {
    val out = FloatArray(1)
    Location.distanceBetween(location.latitude, location.longitude, point.lat, point.lon, out)
    return out[0]
}

private fun nearest(points: List<RecyclingPoint>, location: Location): RecyclingPoint? =
    points.minByOrNull { distanceMeters(location, it) }

private fun distanceText(point: RecyclingPoint, location: Location?): String? {
    location ?: return null
    val meters = distanceMeters(location, point)
    return if (meters < 1000f) "${meters.toInt()} m in a straight line"
    else String.format(Locale.US, "%.1f km in a straight line", meters / 1000f)
}

private fun openRoute(context: Context, point: RecyclingPoint) {
    val uri = Uri.parse("https://www.google.com/maps/dir/?api=1&destination=${point.lat},${point.lon}&travelmode=walking")
    try { context.startActivity(Intent(Intent.ACTION_VIEW, uri)) }
    catch (_: Exception) { android.widget.Toast.makeText(context, "Could not open the route", android.widget.Toast.LENGTH_LONG).show() }
}

private fun hasLocationPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

@Suppress("MissingPermission")
private fun requestLocation(context: Context, callback: (Location?) -> Unit) {
    if (!hasLocationPermission(context)) { callback(null); return }
    val service = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    val provider = when {
        service.isProviderEnabled(LocationManager.GPS_PROVIDER) -> LocationManager.GPS_PROVIDER
        service.isProviderEnabled(LocationManager.NETWORK_PROVIDER) -> LocationManager.NETWORK_PROVIDER
        else -> { callback(null); return }
    }
    if (Build.VERSION.SDK_INT >= 30) {
        service.getCurrentLocation(provider, null as CancellationSignal?, ContextCompat.getMainExecutor(context)) { location ->
            callback(location ?: service.getLastKnownLocation(provider))
        }
    } else {
        callback(service.getLastKnownLocation(provider))
    }
}

private fun kindColor(kind: PointKind): Color = Color(android.graphics.Color.parseColor(
    when (kind) {
        PointKind.Containers -> COLOR_POINT
        PointKind.Deposit -> COLOR_DEPOSIT
        PointKind.Site -> COLOR_SITE
    }
))

@Composable
private fun LegendDot(color: Color, label: String, p: EcoPalette) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).background(color, CircleShape))
        Spacer(Modifier.width(5.dp))
        Text(label, color = p.text, fontSize = 12.sp)
    }
}

@Composable
fun RecyclingMapScreen(p: EcoPalette, onBack: () -> Unit) {
    val context = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current
    val tapRadiusPx = with(LocalDensity.current) { 14.dp.toPx() }
    var points by remember { mutableStateOf(PointsCache.points ?: emptyList()) }
    var loading by remember { mutableStateOf(PointsCache.points == null) }
    var error by remember { mutableStateOf<String?>(null) }
    var selectedFilter by remember { mutableStateOf("all") }
    var selected by remember { mutableStateOf<RecyclingPoint?>(null) }
    var userLocation by remember { mutableStateOf<Location?>(null) }
    var gpsMessage by remember { mutableStateOf<String?>(null) }
    var map by remember { mutableStateOf<MapLibreMap?>(null) }
    var style by remember { mutableStateOf<Style?>(null) }
    val byId = remember(points) { points.associateBy { it.id } }
    val currentById by rememberUpdatedState(byId)
    val mapView = remember(context) { MapLibre.getInstance(context); MapView(context).apply { onCreate(null) } }

    fun onLocation(loc: Location?) {
        userLocation = loc
        gpsMessage = if (loc == null) "Could not get your location. Check GPS." else null
        if (loc != null) map?.animateCamera(CameraUpdateFactory.newLatLngZoom(LatLng(loc.latitude, loc.longitude), 13.0))
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { granted ->
        if (granted.values.any { it }) requestLocation(context, ::onLocation)
        else gpsMessage = "Location permission denied. You can still use the map manually."
    }
    fun locate() {
        if (!hasLocationPermission(context)) {
            permissionLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
        } else requestLocation(context, ::onLocation)
    }

    LaunchedEffect(Unit) {
        if (PointsCache.points != null) return@LaunchedEffect
        try {
            points = withContext(Dispatchers.IO) { loadPoints(context.applicationContext) }
                .also { PointsCache.points = it }
        }
        catch (e: Exception) { error = "Could not load recycling_points.json: ${e.message}" }
        finally { loading = false }
    }
    val visiblePoints = remember(points, selectedFilter) { points.filter { it.matches(selectedFilter) } }

    // Единственное место, где точки попадают на карту: срабатывает, когда готовы И стиль, И данные.
    LaunchedEffect(visiblePoints, style) {
        val s = style ?: return@LaunchedEffect
        val collection = withContext(Dispatchers.Default) { pointsGeoJson(visiblePoints) }
        if (s.isFullyLoaded) s.getSourceAs<GeoJsonSource>(MAP_SOURCE)?.setGeoJson(collection)
    }
    LaunchedEffect(selected, style) {
        val s = style ?: return@LaunchedEffect
        if (!s.isFullyLoaded) return@LaunchedEffect
        (s.getLayer(LAYER_SELECTED) as? CircleLayer)
            ?.setFilter(Expression.eq(Expression.get("id"), Expression.literal(selected?.id ?: "")))
    }

    DisposableEffect(mapView, lifecycle) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> mapView.onStart()
                Lifecycle.Event.ON_RESUME -> mapView.onResume()
                Lifecycle.Event.ON_PAUSE -> mapView.onPause()
                Lifecycle.Event.ON_STOP -> mapView.onStop()
                else -> Unit
            }
        }
        lifecycle.lifecycle.addObserver(observer)
        if (lifecycle.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) mapView.onStart()
        if (lifecycle.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) mapView.onResume()
        onDispose {
            lifecycle.lifecycle.removeObserver(observer)
            style = null
            map = null
            mapView.onPause()
            mapView.onStop()
            mapView.onDestroy()
        }
    }
    LaunchedEffect(mapView) {
        mapView.getMapAsync { ready ->
            map = ready
            ready.cameraPosition = CameraPosition.Builder().target(LatLng(56.88, 24.6)).zoom(6.3).build()
            ready.uiSettings.isRotateGesturesEnabled = false
            ready.setStyle(Style.Builder().fromUri(MAP_STYLE)) { loaded ->
                setupLayers(loaded)
                style = loaded
            }
            ready.addOnMapClickListener { latLng ->
                val pt = ready.projection.toScreenLocation(latLng)
                val area = RectF(pt.x - tapRadiusPx, pt.y - tapRadiusPx, pt.x + tapRadiusPx, pt.y + tapRadiusPx)
                val cluster = ready.queryRenderedFeatures(area, LAYER_CLUSTERS).firstOrNull()
                if (cluster != null) {
                    val source = ready.style?.getSourceAs<GeoJsonSource>(MAP_SOURCE)
                    val zoom = source?.let { runCatching { it.getClusterExpansionZoom(cluster).toDouble() }.getOrNull() }
                        ?: (ready.cameraPosition.zoom + 2)
                    val target = (cluster.geometry() as? Point)?.let { LatLng(it.latitude(), it.longitude()) } ?: latLng
                    ready.animateCamera(CameraUpdateFactory.newLatLngZoom(target, zoom.coerceAtMost(17.0)))
                    return@addOnMapClickListener true
                }
                val id = ready.queryRenderedFeatures(area, LAYER_SINGLE).firstOrNull()?.getStringProperty("id")
                val point = id?.let { currentById[it] }
                if (point != null) { selected = point; true } else { selected = null; false }
            }
        }
    }

    Box(Modifier.fillMaxSize().background(p.background)) {
        AndroidView(factory = { mapView }, modifier = Modifier.fillMaxSize())
        Column(Modifier.fillMaxWidth().statusBarsPadding()) {
            Row(Modifier.fillMaxWidth().background(p.panel).padding(horizontal = 8.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Rounded.Close, contentDescription = "Back", tint = p.text)
                }
                Column(Modifier.weight(1f)) {
                    Text("Recycling map", color = p.text, style = MaterialTheme.typography.titleMedium)
                    Text(
                        if (loading) "Loading points…" else "${visiblePoints.size} points · Latvia",
                        color = p.muted, fontSize = 12.sp
                    )
                }
                if (loading || style == null) {
                    CircularProgressIndicator(Modifier.size(20.dp), color = p.accent, strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                }
                IconButton(onClick = { locate() }) {
                    Icon(Icons.Rounded.MyLocation, contentDescription = "My location", tint = p.accent)
                }
            }
            Row(Modifier.fillMaxWidth().background(p.panel).horizontalScroll(rememberScrollState())
                .padding(start = 12.dp, end = 12.dp, bottom = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                filters.forEach { (key, name) ->
                    FilterChip(selected = key == selectedFilter, onClick = {
                        selectedFilter = key
                        if (selected?.matches(key) != true) selected = null
                    }, label = { Text(name) })
                }
            }
            error?.let { Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.fillMaxWidth().background(p.panel).padding(12.dp)) }
            gpsMessage?.let { Text(it, color = p.text, modifier = Modifier.fillMaxWidth().background(p.panel).padding(12.dp)) }
        }

        val point = selected
        if (point != null) {
            Column(Modifier.align(Alignment.BottomCenter).navigationBarsPadding()
                .fillMaxWidth().padding(12.dp)
                .background(p.panel, RoundedCornerShape(24.dp))
                .padding(18.dp).heightIn(max = 380.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(12.dp).background(kindColor(point.kind), CircleShape))
                    Spacer(Modifier.width(8.dp))
                    Column(Modifier.weight(1f)) {
                        Text(point.store ?: point.kind.title, style = MaterialTheme.typography.titleMedium, color = p.text)
                        if (point.store != null) Text(point.kind.title, color = p.muted, fontSize = 12.sp)
                    }
                    IconButton(onClick = { selected = null }) { Icon(Icons.Rounded.Close, "Close", tint = p.text) }
                }
                Text(point.address, color = p.text)
                distanceText(point, userLocation)?.let { Text(it, color = p.accent, fontWeight = FontWeight.SemiBold) }

                when (point.kind) {
                    PointKind.Deposit -> {
                        Text("Accepts: bottles and cans with the deposit \"D\" mark — plastic, glass, metal",
                            color = p.text, fontSize = 13.sp)
                        point.note?.let { Text(it, color = p.muted, fontSize = 13.sp) }
                    }
                    else -> {
                        val main = MAIN_MATERIALS.filterKeys { it in point.materials }.values
                        if (main.isNotEmpty()) Text("Accepts: " + main.joinToString(), color = p.text, fontSize = 13.sp)
                        if (point.extras.isNotEmpty())
                            Text("Other: " + point.extras.joinToString(), color = p.text, fontSize = 13.sp)
                        if (point.operator.isNotBlank()) Text("Operator: ${point.operator}", color = p.muted, fontSize = 12.sp)
                        point.note?.let { Text("Note: $it", color = p.muted, fontSize = 12.sp) }
                    }
                }
                if (point.hours.isNotEmpty()) {
                    Text("Opening hours", color = p.text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    point.hours.forEach { (days, hours) -> Text("$days: $hours", color = p.muted, fontSize = 12.sp) }
                }
                Button(onClick = { openRoute(context, point) }, modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Rounded.Navigation, null)
                    Spacer(Modifier.width(8.dp))
                    Text("Get directions")
                }
            }
        } else if (points.isNotEmpty()) {
            Column(Modifier.align(Alignment.BottomCenter).navigationBarsPadding().padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Surface(shape = RoundedCornerShape(16.dp), color = p.panel) {
                    Row(Modifier.padding(horizontal = 12.dp, vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        LegendDot(kindColor(PointKind.Containers), "Containers", p)
                        LegendDot(kindColor(PointKind.Deposit), "Deposit", p)
                        LegendDot(kindColor(PointKind.Site), "Sites", p)
                    }
                }
                val location = userLocation
                // ~4000 расчётов расстояния — только при смене фильтра или геопозиции, а не на каждую перерисовку.
                val closest = remember(visiblePoints, location) { location?.let { nearest(visiblePoints, it) } }
                if (closest != null) {
                    Button(onClick = {
                        selected = closest
                        map?.animateCamera(CameraUpdateFactory.newLatLngZoom(LatLng(closest.lat, closest.lon), 15.0))
                    }, modifier = Modifier.fillMaxWidth()) {
                        Text("Nearest point · ${distanceText(closest, location)}")
                    }
                } else if (location == null) {
                    Surface(shape = RoundedCornerShape(16.dp), color = p.panel) {
                        Text("Tap a point or the location icon", modifier = Modifier.padding(12.dp), color = p.text)
                    }
                }
            }
        }
    }
}