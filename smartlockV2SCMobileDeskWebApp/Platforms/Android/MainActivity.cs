using Android.App;
using Android.Content;
using Android.Content.PM;
using Android.OS;
using Android.Runtime; // For Permission attribute and GeneratedEnum
using Plugin.Fingerprint;
using Plugin.LocalNotification;

namespace testdatabase.Platforms.Android
{
    [Activity(Label = "TestDatabase", Icon = "@mipmap/appicon", MainLauncher = true,
        Theme = "@style/Maui.SplashTheme",
        ConfigurationChanges = ConfigChanges.ScreenSize
                             | ConfigChanges.Orientation
                             | ConfigChanges.UiMode
                             | ConfigChanges.ScreenLayout
                             | ConfigChanges.SmallestScreenSize)]
    public class MainActivity : MauiAppCompatActivity
    {
        protected override void OnCreate(Bundle? savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            // Required by Plugin.Fingerprint to access the current activity
            CrossFingerprint.SetCurrentActivityResolver(() => this);

            // Request notification permissions for Android 13+ (Tiramisu)
            if (Build.VERSION.SdkInt >= BuildVersionCodes.Tiramisu)
            {
                if (CheckSelfPermission(global::Android.Manifest.Permission.PostNotifications) != Permission.Granted)
                {
                    RequestPermissions(new string[] { global::Android.Manifest.Permission.PostNotifications }, 101);
                }
            }
        }

        protected override void OnNewIntent(Intent intent)
        {
            base.OnNewIntent(intent);
        }

        public override void OnRequestPermissionsResult(int requestCode, string[] permissions, [GeneratedEnum] Permission[] grantResults)
        {
            base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}
