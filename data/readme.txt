1.Task Specification
Given a question and it's corresponding document, your system should select one or more sentences as answers from the document.

2.Data Format
1).An example:
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t ֥�Ӹ���ʵ�Ӱ��(Chicago International Film Festival)�Ǳ�����ʷ��õ�������Ӱ�ڡ� \t 0
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t ���Ů���ǽ������С�ͼ�ŵĻ��¡����й��� \t 1
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t ֥�Ӹ��Ӱ��ÿ��10�¾ٰ죬��1965���һ���Ӱ������֥�Ӹ���ʵ�Ӱ���ѳ�Ϊ����֪������ȵ�Ӱʢ�ᡣ \t 0
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t �ҹ�������Ӱ��������ı�ġ��ն�������1990���׻�õ�Ӱ����߽�--��������� \t 0
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t ֥�Ӹ��Ӱ����֯���ɵ�Ӱ�����ˡ�������������˶�.������1964�귢������ġ� \t 0
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t ����ּΪ��ͨ����Ӱ��¼�������ֶμ�ǿ��ͬ�Ļ���������֮������͹�ͨ�� \t 0
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t ��ί���ر�󽱣���ͼ�ŵĻ��¡����й��� \t 0
֥�Ӹ���ʵ�Ӱ�����Ů������˭�� \t 2002�꣬����30������Һ͵�����90�ಿ����Ƭ��40�ಿ��Ƭ�μ��˵�Ӱ�ڣ�������������ص�6�����ڡ� \t 0

2).Explanation
A question (the 1st column), question��s corresponding document sentences (the 2nd column), and their answer annotations (the 3rd column) are provided. 
If a document sentence is the correct answer of the question, its annotation will be 1, otherwise its annotation will be 0. 
The three columns will be separated by the symbol ��\t��.
All the dataset file are encoded in UTF-8.

3.Data Statistics
dataset             # of unique questions
train set           7895
development set     878
test set            5997

4.Evaluation Metrics
MRR, MAP, and ACC@1.


5. ����ļ���ʽ

ÿ��ֻ����һ��ʵ������ʾ�����ʾ�ʹ𰸺�ѡ��֮��Ĺ�ϵ��


���磺

0.1534
2.7762
0.0097
15.2345
.
.
.
.
.

���۹��߻ᰴ����ֵ��С������ÿ���ʾ�Ľ����������
