2019_12_23
stream_patchとdelayにenable信号を入れる予定。
それに伴う各使用モジュールの調整を。
layer,maxpoolingにも導入
delay   レイテンシなし
stream  レイテンシ１？
layer   レイテンシなし？
maxpool レイテンシなし
上記四つとりあえず変更
enable 信号を出すのはmax_poolingとunpoolingかな？
→ 全モジュールが出します。Netはenableをバッファリングさせて出力...?
 → やっぱりmaxpoolingとunpoolingだけでよさそう、後段二つに出力（タイミング変える？）
   底のunpoolingだけちょっと怪しい、maxpooling→ unpooling
   Netのレイテンシを考えなきゃならんしやっぱり各モジュールで出力かなぁ
   Netが前段に頼らずにenable信号を出せればいいんですよ（できんのかそんなの）
前段→ unpoolingの実装が悩ましいわね

全モジュールが前段のenableを受けて動作すれば大丈夫なはず
unpoolingだけ前段のenableを元に四倍動く&レベルで動きを変える

max_poolingのレジスタ、最上位bitの左右なんかよくわかんないから確認して
layerのLATENCYめちゃわかりずらいよ・・・
→ 20200110 修正,全然難解じゃない

coordajusterのLATENCYいじらなくて大丈夫っぽい。。。？？？
計算だからいらない

coord ajust の動作が怪しい！！！
1クロックに1画素進む前提での計算なため、
h_cntとv_cntを既に調整してる場合ズレが起きそうだ・・・
計算し直すか、delay(enable == 1)を使うことになりそうだ
