using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace 图片标注v1
{
    class picture
    {
        public List<String> list ;
        public picture()                      //7.调用此构造函数

        {

            list = new List<string>();

        }
        public void getlist()
        {
            

            //遍历文件夹
            DirectoryInfo theFolder = new DirectoryInfo("imgs");
            FileInfo[] thefileInfo = theFolder.GetFiles("*.*", SearchOption.TopDirectoryOnly);
            foreach (FileInfo NextFile in thefileInfo)  //遍历文件
                    list.Add(NextFile.ToString());
        }
    }
}
