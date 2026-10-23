package com.ezymind.chehra;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Base64;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.Locale;

/** Hosts the Chehra web app, which does all the vision on the device. Adds what a WebView lacks: voice and saving. */
public class MainActivity extends Activity {
    private static final String URL = "https://pallavsharmaofficial.github.io/chehra/";
    private WebView web;
    private PermissionRequest pending;
    private TextToSpeech tts;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        tts = new TextToSpeech(this, s -> { if (s == TextToSpeech.SUCCESS) tts.setLanguage(new Locale("en", "IN")); });

        web = new WebView(this);
        setContentView(web);
        WebSettings ws = web.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setMediaPlaybackRequiresUserGesture(false);
        web.addJavascriptInterface(new Bridge(), "ChehraAndroid");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest r) {
                Uri u = r.getUrl();
                if (u.getPath() != null && u.getPath().startsWith("/chehra") && "pallavsharmaofficial.github.io".equals(u.getHost())) return false;
                startActivity(new Intent(Intent.ACTION_VIEW, u));
                return true;
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest r) {
                runOnUiThread(() -> {
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) r.grant(r.getResources());
                    else { pending = r; requestPermissions(new String[]{Manifest.permission.CAMERA}, 1); }
                });
            }
        });
        if (state != null) web.restoreState(state); else web.loadUrl(URL);
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (pending == null) return;
        if (res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED) pending.grant(pending.getResources()); else pending.deny();
        pending = null;
    }

    @Override protected void onSaveInstanceState(Bundle out) { super.onSaveInstanceState(out); web.saveState(out); }
    @Override public void onBackPressed() { if (web.canGoBack()) web.goBack(); else super.onBackPressed(); }
    @Override protected void onDestroy() { if (tts != null) tts.shutdown(); web.destroy(); super.onDestroy(); }

    private class Bridge {
        @JavascriptInterface
        public void speak(String text) { if (tts != null) tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "chehra"); }

        @JavascriptInterface
        public boolean saveImage(String base64Jpeg) {
            try {
                byte[] data = Base64.decode(base64Jpeg, Base64.DEFAULT);
                String name = "chehra-fit-" + System.currentTimeMillis() + ".jpg";
                if (Build.VERSION.SDK_INT >= 29) {
                    ContentValues v = new ContentValues();
                    v.put(MediaStore.Images.Media.DISPLAY_NAME, name);
                    v.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                    v.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Chehra");
                    Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v);
                    if (uri == null) return false;
                    try (OutputStream o = getContentResolver().openOutputStream(uri)) { o.write(data); }
                } else {
                    File dir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "Chehra");
                    if (!dir.exists() && !dir.mkdirs()) return false;
                    try (OutputStream o = new FileOutputStream(new File(dir, name))) { o.write(data); }
                }
                return true;
            } catch (Exception e) { return false; }
        }
    }
}
