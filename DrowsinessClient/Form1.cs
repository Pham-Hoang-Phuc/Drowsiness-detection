using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace DrowsinessClient
{
    public partial class Form1 : Form
    {
        private TcpClient? client;
        private NetworkStream? stream;
        private Thread? receiveThread;
        private bool running = false;
        private PictureBox picVideo = null!;
        private Label lblStats = null!;
        private Button btnConnect = null!;

        private int blinks = 0;
        private float microsleeps = 0f;
        private int yawns = 0;
        private float yawnDuration = 0f;
        private double processingTime = 0;
        private float eyesClosedDuration = 0f;
        private bool drowsy = false;

        private System.Media.SoundPlayer alertSound = new System.Media.SoundPlayer();

        public Form1()
        {
            InitializeComponent();
            SetupUI();
        }

        private void SetupUI()
        {
            this.Text = "Drowsiness Detection - Python + GPU";
            this.Size = new Size(1100, 700);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(20, 25, 35);
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;

            picVideo = new PictureBox
            {
                Size = new Size(640, 480),
                Location = new Point(20, 20),
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.StretchImage,
                BackColor = Color.Black
            };

            lblStats = new Label
            {
                Location = new Point(680, 20),
                Size = new Size(380, 480),
                Font = new Font("Consolas", 11, FontStyle.Bold),
                ForeColor = Color.Cyan,
                BackColor = Color.FromArgb(30, 35, 45),
                Padding = new Padding(15),
                TextAlign = ContentAlignment.TopLeft
            };

            btnConnect = new Button
            {
                Text = "CONNECT TO SERVER",
                Location = new Point(680, 520),
                Size = new Size(180, 60),
                BackColor = Color.FromArgb(0, 170, 255),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 12, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat
            };
            btnConnect.Click += (s, e) => ToggleConnection();

            Controls.AddRange(new Control[] { picVideo, lblStats, btnConnect });

            try
            {
                alertSound.SoundLocation = "alert.wav";
                alertSound.LoadAsync();
            }
            catch { }

            UpdateStats();
        }

        private void ToggleConnection()
        {
            if (running) Disconnect();
            else ConnectToServer();
        }

        private void ConnectToServer()
        {
            if (running) return;
            try
            {
                client = new TcpClient("127.0.0.1", 9001);
                stream = client.GetStream();
                running = true;
                receiveThread = new Thread(ReceiveLoop);
                receiveThread.IsBackground = true;
                receiveThread.Start();

                btnConnect.Text = "DISCONNECT";
                btnConnect.BackColor = Color.LimeGreen;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Lỗi kết nối:\n" + ex.Message, "Lỗi", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void Disconnect()
        {
            running = false;
            stream?.Close();
            client?.Close();
            receiveThread?.Join(1000);
            btnConnect.Text = "CONNECT TO SERVER";
            btnConnect.BackColor = Color.FromArgb(0, 170, 255);
            lblStats.Text = "Disconnected...";
            picVideo.Image = null;
            alertSound.Stop();
        }

        private void ReceiveLoop()
        {
            var lenBuf = new byte[4];
            while (running && stream != null)
            {
                try
                {
                    if (stream.Read(lenBuf, 0, 4) != 4) break;
                    int imgLen = IPAddress.NetworkToHostOrder(BitConverter.ToInt32(lenBuf, 0));
                    var imgBuf = new byte[imgLen];
                    int read = 0;
                    while (read < imgLen && running)
                        read += stream.Read(imgBuf, read, imgLen - read);

                    if (stream.Read(lenBuf, 0, 4) != 4) break;
                    int jsonLen = IPAddress.NetworkToHostOrder(BitConverter.ToInt32(lenBuf, 0));
                    var jsonBuf = new byte[jsonLen];
                    read = 0;
                    while (read < jsonLen && running)
                        read += stream.Read(jsonBuf, read, jsonLen - read);

                    string json = Encoding.UTF8.GetString(jsonBuf);
                    var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(json) ?? new();

                    blinks = GetInt(data, "blinks");
                    microsleeps = GetFloat(data, "microsleeps");
                    yawns = GetInt(data, "yawns");
                    yawnDuration = GetFloat(data, "yawn_duration");
                    processingTime = GetDouble(data, "processing_time");
                    eyesClosedDuration = GetFloat(data, "eyes_closed_duration");
                    drowsy = GetBool(data, "drowsy");
                    Console.WriteLine($"Received drowsy: {drowsy}");
                    
                    using var ms = new MemoryStream(imgBuf);
                    var bmp = new Bitmap(ms);

                    Invoke((Action)(() =>
                    {
                        picVideo.Image?.Dispose();
                        picVideo.Image = (Bitmap)bmp.Clone();

                        //CẢNH BÁO BUỒN NGỦ 
                        if (drowsy)
                        {
                            using var g = Graphics.FromImage(picVideo.Image);
                            string warning = $"WARNING!\nThe driver is drowsy";
                            var font = new Font("Segoe UI", 22, FontStyle.Bold);
                            var size = g.MeasureString(warning, font);
                            var rect = new RectangleF(20, 80, size.Width + 40, size.Height + 20);

                            g.FillRectangle(new SolidBrush(Color.FromArgb(230, 255, 30, 30)), rect);
                            g.DrawString(warning, font, Brushes.White, 40, 90);

                            try { alertSound.PlayLooping(); }
                            catch { }
                        }
                        else
                        {
                            alertSound.Stop();
                        }

                        UpdateStats();
                    }));
                }
                catch { break; }
            }
            if (running) Invoke((Action)Disconnect);
        }

        private void UpdateStats()
        {
            string alert = "";
            if (drowsy) alert = "WARNING: DROWSY!\n";
            else if (yawnDuration > 2) alert = "LONG YAWN\n";

            lblStats.Text =
                "PYTHON + GPU SERVER\n" +
                "━━━━━━━━━━━━━━━━━━━━━━\n" +
                alert +
                $"Blinks: {blinks}\n" +
                $"Microsleeps: {microsleeps:F2} s\n" +
                $"Yawns: {yawns}\n" +
                $"Yawn Duration: {yawnDuration:F2} s\n";
        }

        private int GetInt(Dictionary<string, object> d, string k) => d.TryGetValue(k, out var v) ? Convert.ToInt32(v) : 0;
        private float GetFloat(Dictionary<string, object> d, string k) => d.TryGetValue(k, out var v) ? Convert.ToSingle(v) : 0f;
        private double GetDouble(Dictionary<string, object> d, string k) => d.TryGetValue(k, out var v) ? Convert.ToDouble(v) : 0;
        private bool GetBool(Dictionary<string, object> d, string k) => d.TryGetValue(k, out var v) ? Convert.ToBoolean(v) : false;

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            Disconnect();
            base.OnFormClosing(e);
        }
    }
}