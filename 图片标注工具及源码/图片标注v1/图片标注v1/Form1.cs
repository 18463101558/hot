using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace 图片标注v1
{
    public partial class Form1 : Form
    {
        private List<String> list;
        public Form1()
        {
            InitializeComponent();

            picture data = new picture();
            data.getlist();//读取图片列表
            list = data.list;
            if (list.Count == 0)
            {
                System.Environment.Exit(0);//木有图片，那么直接退出
            }
            this.pictureBox1.Load("imgs/" + list[0].ToString());//首先展示第一张图片
            this.textBox1.Text = list[0].ToString();//展示一些画面
            if (File.Exists(@"data.csv"))
            {
                //如果存在则删除
                File.Delete(@"data.csv");
            }

        }
        private void saveexecl(string path,int classification)
        {
            if (classification == 1) {
                    FileStream fs = new FileStream("data.csv", System.IO.FileMode.Append, System.IO.FileAccess.Write);
                    StreamWriter sw = new StreamWriter(fs);
                    string temp = path + ',' + '1';
                    sw.WriteLine(temp);
                    sw.Close();
                    fs.Close();
            }
            else {
                FileStream fs = new FileStream("data.csv", System.IO.FileMode.Append, System.IO.FileAccess.Write);
                StreamWriter sw = new StreamWriter(fs);
                string temp = path + ',' + '0';
                sw.WriteLine(temp);
                sw.Close();
                fs.Close();
            }
            
        }
        private void button1_Click(object sender, EventArgs e)
        {
            saveexecl(list[0].ToString(),1);
            list.RemoveAt(0);//当前图片展示完毕，准备下一张图片
            if (list.Count == 0)
            {
                MessageBox.Show("恭喜恭喜，数据已经标注完毕");
                System.Environment.Exit(0);
            }
            this.pictureBox1.Load("imgs/" + list[0].ToString());//首先展示第一张图片
            this.textBox1.Text = list[0].ToString();
        }
        private void button2_Click(object sender, EventArgs e)
        {
            saveexecl(list[0].ToString(),0);
            list.RemoveAt(0);//当前图片展示完毕，准备下一张图片
            if (list.Count == 0)
            {
                MessageBox.Show("恭喜恭喜，数据已经标注完毕");
                System.Environment.Exit(0);
            }
            this.pictureBox1.Load("imgs/" + list[0].ToString());//首先展示第一张图片
            this.textBox1.Text = list[0].ToString();
        }
    }
}
