#!/usr/bin/env python3
"""Rebuild the evaluated AFR1 archive from the pinned generation-6 base.

This is the public, self-contained encode path.  It decodes the token field from
the supplied base archive, then runs the real RC64 encoder through the five
lossless stages in order: FX5, DX2, GB1, LB1, and AFR1.  Every generated payload
is retained and every stage is checked against its receipt-pinned SHA-256.

Place the pinned base beside this file as ``base_archive.zip`` or pass
``--base-archive PATH``.  The base is an input and is deliberately not embedded
in this tree; this encoder never performs a network request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


HERE = Path(__file__).resolve().parent
_ACTIVE_VENDOR: Path | None = None


EMBEDDED_SOURCES: dict[str, tuple[str, bytes]] = {
    'fx2_afr1.py': (
        '6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064',
        (
            b'c-q}PYjfK=cHj9cu)34Ilp`sYVmpab-Kk?MiM#Qml6<pwJg!WMl+3k6DkNoFyWN?7?B`B@VSh=_0{{V%a_n@s(`n}3L?S@o;NZM*'
            b'a8#?+e19>TK6FG=WH*6WCh^!Chx0I7H&4!vdS8V15T^Sg2`4u}A|~D{_2yz0#ET$H*5W0-G#K<=9h{$tH1lRb%W=MbJ{XGO=i`Am'
            b'I64}L;hDI;=4sm3*9~!SdL+Jn{!R1_4nJd$<Kgkyses-e`@J5#ZgfU??gIZ2&;<N95AT9~;fZPZ5cthy97dVA55n0^7WmGm-q}fS'
            b'*grlLi@=LSJQcC;r(&_1XW?=l21zQiIKCCl-M#JB_8aj5x=d-pXm5KP+HQjgMh@n_@G{{!<1lN6eh_8h1Tf(Tlh{W9^KLC6j<Zz6'
            b'_Yo}nwD-s1==kHP-y2+<^p0Rfk+;D9_p!5z0AUHhD8ejQqz&OkxKdcqBKBAF0GBq7vm1mN;8}>}e3cSplQ_zPhpg#`=>+ft8-b{B'
            b'uqj>!iI>HR@X}itc^W4GPmlU6g2{~+h3P`TR?_IRAPQh-MG{N`z#;4}EC#-Lei+T1IGV4;!IysT;OMvJQST6Nek9WMVjRx_*fbX2'
            b'e2%b%8IF|z#Lag5t(YbO?8cviBo<lXMd>2Ua399&EP&UdSj+=2i4X_g7#5k{gv({%Q_CG7Hg5U=6K*=J5)c$+Koad1&{Izw9$X9#'
            b'zKCJ(<owIQuqVDA48+ChQLo>;7yxW8>~OkDcre(jG+hC%>n2IHM(NVaLT^5r2Y107_5)`DT42O}>3tm1He8&GB~a5G_7SiEC>-9v'
            b'?%#VUPyoIWBzG{(9c;x-98LnU@Bnf5VRjR*GQnT~3TaFx5fLY_E*eO$*}3=bf@uPqi3n`J+t&P*#H(d0e8fW}#%mxXdNc|nKX_=Z'
            b'-Xa?zn4a)M*u0FGBXEdN0M2O$h;`MXJFv=ryN<hz#KJ4Z?GnL*H_7Jfea8{-ho*2J8e&}+UEw~6-;2H1b>Q)i*b?rVzd8;8yb$*{'
            b'AP|@q31nbOBn2<Rf2L^wFvRnDe4irpUIK^#fq|6fWhh4gN?ez71LO)gIS>)D>o{H|hzW0wB)HbhCBcMA0UIG~6+tQp8Q4F~fQX@k'
            b'lP-WH3CkbjgrK`PZ}yKreI6PUTwl8nFWa!nX1jiUE!WPA6~F7^-}`_I!ocr&{cXg-mb<2}O+W`UikZ8_FKu}FALxrYPS+eq{@E^#'
            b'Vve)7tH*JGwV0<QIpDejWY6pCwT5#WcD31*8v%WLe>^xGe)&zw2S3chI70q+7!NrHm~<4<KvH_QYD-El6Gs3(Pb6mnJJ5KrL35Bt'
            b'UIeTTTk-B~Ywz98?u!O|X(lLnjYsof%3oh^w_lLJh?)RL<fqOuPHCzXrr6p-*3j?g8%A6=zxUSg5BzuMg>#Y|@MX1ZZEZ16-QUFX'
            b'z*zxbC31vsE6)c3l#?O?o`U#=5!b?nnHaJ+t=BsMru(wfXm;xK{p|~s8iSZ#0pScQ&5*xA?xVVZ5dkg@Awqu`C&-9VkOJSvK}r%R'
            b'3DOKOMoj7CT(B<!639s9Our#W(wT><0N91u6BG<UdVQS+^J(ig0(Al#z5uD-Q2gKk0!&@W10a8Zc3}Z+V37m^Om=!U6l>s5#ayUM'
            b'l5o5N>FfZ^Hz>ADF4gVWF!Pcz>Uc)E!Pmh9peO|7+~4p55LZBY3Ux5?eVl+UL+~bXn!Tr9xE^Xl;1)oKDN=swP0@_#w8WRQLy-5M'
            b'&rZ%h^-g=o!{5+MI1nGsK<Xd#e-lT&VefEw)+g=2M#0hX5v}p_!LMjcu<8pyJ<<ZC^cT~>%YdFyOQxdc$6~QI_EUH>P+}y{3C#0;'
            b'@96jt&6HZpIgO*Iyhql}8Nk)A5xU|O5eDiYoK8XgMw#<%1>}LmuQWeUXMo%%aE6fSRhn9W;uv`?Bec&J-h;GBK<zI>P~Z1q8fc9v'
            b'H(zlV@<tly9XH_xfL0M01&bI=ixf8}j2ur)0w72*qd*g*DM6a%`=fz5!U&*&peO1|KU_51JRmW>muPiL8;_<4kwTcJ0n$b^0ptJ$'
            b'B9&cVkM#H>&=!jrO%_l#q_}}c7)eLRgW>V%;ZRKG9*yEHVyS&|aUq3i4zGjjLMa<BmMdf(H0qK7L?Wp;wLriq5HN^Zq~CZJM=(eh'
            b'OOrJ4L5KoUXbT-QeSqd(v$X<dv3MoWFdl&M1V-;*c<`wwToSSfC~PX3%rqA8uo$mG*it|ANqXFXb%0{%Bz!>91u(}U$QLj#7UO_~'
            b'&T$5-im)%3zOY5(&`X)oL0I2`@sLW?KJ1H-j7SGC5s^*>(p%udQk1LPt#@x=zkzHPDBEW!dK67&$toia1D$s9NfMxa<s<>y*a#b2'
            b's0uO|Vd;el@^V9r14MivMIQGsm3#)2f!3xL&M8>0YbjyQ5B?e8Nblt<ia!`@r2lh~39Af}pHp%6_0&1L7@l7YanlC$1`wSXkmV|R'
            b'yl^CBXa)i2`QDODxrMjnov)ENqA`I2=8oLvz^~!Pb1O*T6&M?9*d<UoV9tVMaE4Md0w2%>MBGWRO65jKyANiXfSg5qNL3#}Phvj!'
            b'pQJnL5im`*=r!1OTt3VCB-y#XhGhbze-6;a==_V@ynWRHTLwCV#sNL3Y-<1wJRI~+9OR`o*csnOfT!zgg9U-JQ4%5-fIv?H?78QI'
            b'@NNighY(0CrNRt`0x3F?OB$olpT{%S1?Uq2PC~XMGGM(%ZlN&VBI1a94|x}PMqpDECkuKr9>YuM9MDF9_REqWo7}+Kx;RS{eK&yp'
            b'D+)&5B7_A=P#WJb9zN9&wImKFte<gLq+}f&ob<#W$g#akZ`aLX5>G)6-mzdo9T>(IRl_zmV0ci4xn}StlNGp<q<KLH!VJ9g;{aj0'
            b'zLw~=H0t&Bas&b#Ho+klCYAzQi#;zga&Vb$JVrMI2EGXioS+Wkgj&dx)l@0I)g_i;Jnh%|D8G3D<nRKE=HLJd;PnaUjQLtx@$+z;'
            b'@G_)iKzhE9L0+=v1P=pDt1Jw1b+sn_p&Zf#+(PuiCrE!N`y|<p12De813NiB9UX$=9>V-YS*8<iTCnZbHcug&*El^eIXp*oXF1^S'
            b'm2wJ_X+Y+zu@VH}2C%8{Cju}DNq7?i0Uh8+IZU7AYf0CLcRr>G;q`lvP-LGTAPLOT!o%gU5JFof)gB6Zy$B%S2*HF+ZNk4`qJpRi'
            b'wXpVsS%Ur|P8LQ$K;r;)Hu>A2bwC^jIQ>m9U&1Erv`}#jQN^J#1e#>n`{NLG-r?D=z5Z`f?vVIXkpUngAEi7ApnLp_6-wkcXJB=t'
            b'fzJYV&5iwo(?5X!JO!DJ5*sW_z!})*DBhd#66pcW0boXOsH2QkjC8hPspuzoxH!;Uh-Q!@S;$(EiNg;ZP&+f=85DfM9SJ8;NCA9h'
            b'(}2L3k-Lk8F-F7LUwc5{GLj>OHQDzL84`j_5fUkD#2sRdLkk$EieOK`iU8M`Bnd%mz{NVB;fNfxx0axgwFeL8cZ5{u0WnOJxvO92'
            b'(e6?5u`B^V5{z<4iSMP3imadwHIsoA80IXJ{5BZ&FAj$n{hpX4F_|n9Ow_yHe4U1=G%_3=oN+zc!Rd;lE<AF|!WO*<uxTa%`Tq6y'
            b'7)_z&QCx*AsyL*$h+_v??!5Vmjm<n(Ly})|0#i74K-yMngC+!C^aG%gs7bC$7)?=%GlFn1Fmn}=`0NFNv<5khPRBf4vgbpLO1=Z>'
            b'4Mz&xSbI65;N_%Jioa1WhZzxIC}VEOq0|G7f0QVvy}<zBzKAgTHOp0t9&#U1bbSpf1H&rU*C5CF13c;INB9Y(*fPEcAY$~zx3=(8'
            b'q=GzQieQ0%al^h1Y+;5XA4&P%o8P8pY!4Lp!C*)hD9C;?5P%4EsE)apV|h|#<&>oKbi;Ut?qW1+ILwKx>i~BNG75}zef<)2{}$+v'
            b'1cRgLw<JUL<DlVu)<?$%rbfpGxu4yLWW`uCh}MX}f{k!57@;XG*QtnvlJqyFH4&CSN`o1RZGW^xPeMvXP`#jYYPFgSF^xvk75ZqS'
            b'kq8${Fun!#BZQ=mBcDa9#gb?wTFUoYu&E>1I`sG^BNB@vi4i0de$Az5&B^f}d;O7=sDlPQ``W{J(FpO<8=al@_~}OsEuUj#`HLL+'
            b'1PA6u`uNdXg!45&V95omyH{OkBSv8VO*4KV4Vw{yJW8-TpvS4pC4iAPLT6$V;G@As|Ga-Z=#35rhdiM2iAP_0zxKY!#hhN8j1IpX'
            b'fVz?o&W?@-qtCEH<Lj5Rul()&^8qXzl#E>TdH?LgsCO=3?O&W)V_%Fu9G}Y1pD-XdKz@As8a_hXp>Bd59I$n4J;&jk(FklqAcL;B'
            b'q&I6uKtO_2V>oJq&&TKIy(8SykH??%({s?D2VZuIodJJp^VKtJcKz_7bAq10!9$qRW~+wjLuWLljUT08{H5k|MR}}{q2<bPJ`-`l'
            b'LSDb!alRfO4L`%@-FC-uegXy8)R7akfTh2xLntoBaGOv3<2W2EADw;tIOq)r`yyK{=fNfa9iq7be_b_1tJS)~Sf|@;H^ep=k}eou'
            b'%{G2$s}D{7R7d7?{f7q7cir^+We7cY#nciC(Uue>&KZWmxu@0*MhKauDMc5{F>gE6=zK5KNWsWF#MjxigO9=tJ|4{H4h8i*X@;XO'
            b'6|ZJ&<zHXh0XV51DWs2P0(wzka7?G+Bn+a-+9B1nAghi}B9us`@E{6$mUEx_XBX$hMz95M@fj+m*|@#WewB2q+wZoUpi7oyo`k4w'
            b'=lWQIu>*PoF`>O}dEl`KQDbU6<5;9sX+$~zkwJ7cBtISxhsUR%guB~rb#}!Ev_JhOx>Vi_LwoSn_D<_H#tE9>r~@6<(d)ot1lk3H'
            b'7~c_lJFU((=$gIV)|)piT4{j;VYz#_-0SOYa_!JYDTHS*NcmpG=p0dO3*Fb3!o7*_VH4<>!H(}ZrT7jEnIvm7P(zjn*m-r->M7Es'
            b'N4^bMG`kFES+XF5J`+?rJX}$r1EAEwDXAfyX9cOiRP`uk05I-QM6AvJ8wcqe6{JHe<(IDLs75(ilNGU!ywM^JVUx$G*qw3vou><#'
            b'T-cx_m1N9n0WtQ$AYm1pK?@B?3T$uFnn2&DGQ8nf;SG*U7*P!s)1Y_;DCQ;wCVUlt0J$_J+k+wzjxz4OJUU=SC(v|%v;BfW=3}M6'
            b'Tl`Em3?DK}S~k?izCgOVgsu%?$kurLC*bxgl&pVYjSx?#wS5w<XlB)=p7;h#gO3&dK!SOTvG0lLzL;Ke^Ofp6nWrQ5f}z;KtlRK9'
            b'yimI`nt^{xgO0&0QG+__sH$K~m^0`;)=Y}NHcrRynJcNB*k0E2z;KNf7vFqo16hAEloWFQ9-I|I<LRJQyi5?$%ufLIugGd&Nd!0X'
            b'+?D;G7{uRHo`IS{Q<5M+i@;47tp|}qCoVejjS;ypHDfya9PCnW5oeGD6N^=fM_j<i{%R6fPMQ(?kRrzqrlM#qxgctODKgO2K4Tvx'
            b't{(7b74-(C9nbSb2o;($jSg)vt)h86xkaM`2Sg8EhPKcWjZ)}d3QDiElPKjQMWp{Qv|@rF@W_IXWq1ITZgw98By-X{ZZ3nm819u&'
            b'veceq?1{WWmW?!+lqX;=&_OS@+{ennLfMbUBFg*5v=0oM@hOo!JM;B10u9Kq4YXu95=^068TV6RMG$x`g>3rw>Hd?sjf`LV2=WfM'
            b')3tZG&OML>q(!(h0*hn`UTok_>-(BRz$IEGk;NWG!+}^pjl%44VKB#vn4@Ao#r%JR`Mx7u^2u3vJawf{5Co5{R8A%#{Tod79ZVEw'
            b'RtTtG?+u#N-RFYvJxmuCNRvE~9#Bfei8ucqlpP+2&NbT%mHDW%5sD|H4<1c!+V46ejQ>{c8@#9bsP~AmTB{|7Tiw6ZtRYZQ!RLlO'
            b'aQ&|a7or2P^L9l36(H#ChFQ5Ap~3J}Xd(L54G<M)K%jQsS_3soU|2Bfr;p*K1O~o|di#4zQ``I9b=jaim6snG_Ax%Tkf_K1i~dBi'
            b'+C~f_*2JmZT5s6zTBs@8k>)^F*Wx&3j9BJTwV7lfWy^{lnSaV7+kFg3{>moMGf-F3J{6{Pfo(OUWz|(gN=er|0Vn0@D&u(OpVBlN'
            b'5kyegN~pZxO5;_n*?xJ$&`PF#f3?+yvW<fBg6(OZvf7d^`cc@*9>^xT-7xXv(3Z4Od3!qY|00&knr!3X3Kp=v)2y?*_d{g4Y2Bq+'
            b'%3WZ}=aA?#z<i8$H<8f~fvH4|A?~H=s~sCV+*|3|*s*~Pe$IWBE2kZv7!!c{ufgw#Zp|Zpa4P(kGUN3ju(ZbQV~)Dj=X)YEcY;?+'
            b'XZ*;fye;GVhuz_~rJ-JmXxUnXQMgzwT&+{ZFTaS_4e_S_IQ8KQH5ET~k!1#X@(Q2eVf-Cds3p_$HnS`Me_(q{ZTRPz;hOKzMQMt9'
            b'>B-+0t>g_5THAX+gx4MTzrFW&$K#Ie&phJJJch~k-ZM}Mu1I_L`MCU_66((1l2|LB{s{Uv4piEfC-kh^uIer0Z7ZY0**L-fEyjAb'
            b'fUc&8=ZfwB&%{u`#b!z{f)@X~#86>%DYFF(FAhSNj^^DBsT%c7ab?Pz(+QoMPTy5jS7-jou<}<9j;Cp$Y`^Z$@pSt0^)+WEMx{8g'
            b'K+}%W8_kS8UbPWj$18Qgw^f+%Nmq58{loFF`SGmZgnx~UUKLc95qz1hN4G0D;0^f7JWk9sLd0xgM!~&tQ;_pkC2W{1ol*y|M==?F'
            b'X53sbvYqgFGofr|x&*+-`LKYwx6b3uI=V6%0cB*PQO;I)<t3*AdCLfGXZ>s||7c)-ACK(7=%vFpy9Xr(j@Y)|7h^m`=!%ct+;jxC'
            b'wr=meWR}XK6*S=StS%4qP*gh%(5@zX-sS$SWxRB2s<GA(ZcQe7kN$*YSZmblIipy90*nM8TjN;mFjy@D7-}kpj(>hq*@$?udFRb3'
            b'>5_BDVHnEL#G5F5W`8^!$dogA;Ex9=X`U~vhV)~~RKPq3ix1G@0Me3Iq2#1tn&*gdZlA{oGn5^k4~a(?lS#15G?}Pj)sYCMTOgcx'
            b'Zx6U-3N!?`pd9a^q+%Y-n?qB?cP^Vvu~=n%bYD1WQOR|)c|6AJxqLgrOK{iZxebqZAFt*<x=pw2L@{GXDrOf=9U}wdO@cHv^5HpU'
            b'&u`@<CJ#T#CtSLV;gnUmi@IMprHxSJv~LN=#KZi8UomI1mn3oG)_}8N5p>Ngkf$=H=%|^5m!^oZuDIl@eJGl8n#Or)=1}^!S0+gE'
            b'<?lc<t*s);l*z#tf03{LwN|$WG}%DTtI+Y}X)F<nQyGK)F2O1DEm|!>In-NP+>l&pXh^$du5H**c9|6$T2Ppy<1IZyK3p0(fi7P8'
            b'hPMc-m-q#=KW%_VX|o!HD|<L=@)33J+7Q|OqXJPi__2IS7k;}M#g|s!tKvv3DMz5Ha5a^~gQ}<JbTm2|uO_$9+2F7f)^^)?dMR_b'
            b'DYKtBjk0BwMuZ;Q`zlX_?&W>xeZn}gXUY>RT^<uqbt0BZ`=%3V{46KIJLkf~+n#i?=^29&{WNEB><ca!!1oBIUWkH74Jb49F&7)f'
            b'k_|%6{zy@n=LY*umlUP~&_(+N%E+cF5rEDS+t3RsU`pI((oSI~7;(4<U_xOAcIvw2EyBdN$hF3)1NHjZdKq-#MVNSRCucvLStY+U'
            b'R0cTUECStuDc`PKPu=E26QIleh92UIp`%G$N_2I3%&k%~t*OQUONQ=NBEU+nd8!yl(;ycD#-J<$EclI9#Zty2%VtQ!KL?73@Vn$7'
            b'6T(a5-K+9Mrquy&qS@gVpO@j=Hxr7*zmgGxr;-*nO!-oSmq09Sg1N3xjg3ryIHXfH;(ron5vB+(p&3{L1SY3Qi^JZRFFKz@o4^ga'
            b't7lx$pNF@B#bwOjKc|XOQ<iv~wXT_ruik_i+8J08gAyJ{R)npfRgeHUyK$LIC4+fsGfX#JHbX<i7{(ZI02uMPz(P^HVTvH$BgPhk'
            b')hN~MAYT|a8AH<5Mp!NjM3oau_hsZONU;Il!iXjTCziao@rsw0j1i+)$w|&a$<*Ki*;itgZo7c^q9hGAZosCBCv<}&n9oxSjdY&?'
            b'6F>2y45yqVbQuRsD<99VCCw7x`c|4pNpPH6(kfHCrgIPm)75<LmZe8oMQOoN5*0=HWzeg`>!?nc*2QVuep|c}{f@ruy(zWP@I`ox'
            b'{L*nsQWdNOa02%cSajbLZ%y6-2lexZ4Wq+VF1*0oIZ~JH$+aU@!zrJks+>XCSKg*y$^&iXa<fmKFkXsrWhQAOPawrKNg4`<OmDm;'
            b'x{H@P0N521t}$yah|@yJ0W;Z|s`cocX}(w<sTv=A6P%80^BfAg56g81fKu_AVL-V_A!R;;i4AL9OQKpudFm_drjaGB$5Ygq@|AMB'
            b'6HnzojOqi@Q1{~rR<(#`Efe?p^8=dRfL`%JQ0l#ISgVFtuL><HRo7I%sA}!x`a07`3+3x0YByi%4-AL8!`9&GJkG$^RQdRp0xt|`'
            b'E7AxN@gQCb|G|Eac)*%Y{@QvjbLBt-!JEHy#dcl%o<9ErV~fA8jixm;@l^*qK$K@p69(`W7<6t3ko59qL{qdeGR9XUUgn)1s1rZN'
            b'KGv`rv6qMwG)E-JzZZG+j7vSa9mbMdtY3vGDHYf)j4HM2WR;)`Y>d_d?+bRw)f-NiC5qW0WmaX5{ztqj5Ke%OZ(=`}Lpt6`PP<zq'
            b'NK+IUL)wqW9vUb3g84iTZsgA_S;dL&?1tXR+YRb+*E#@_BMXAl@qcoSbbBy%^MVK<h*~D|QHYY=PuJCK)ir>;<ojp)S4cy-Kx?I|'
            b'g^T5_y4waTMeC8tPWp2TT1vg;7UVl?zTGf`lN!pGqTMJ(i;?i~l8oAMo3=NosSO)!)UuF!{PeATZ1;|+(8<K8@A91628Nm$TE$1C'
            b'3f-6Y5AgnU9!xA00^z;p%d~np!|P+SP86z)yw5K%@aUv5bkUK#q=*_(#nSan5H2XuRPY$LeG5tcc@Y&WI8v%Ua){;@TX4!|1rlxE'
            b';Ef)2?;l~qs#-eeO5&L+gDH-;%P5fczoT>g(260M+(JpZQD!L$^DdtnyhS?xvVHX(J%QRG!CYe3esTM%pc__ck)v4pS)FiH?KSeS'
            b'25^NDjpba@OIPAwyJ{X#76tj+<&pCx_==#rY8X=@l!R{u(<|myBQ>gy1xcyc8Gj?X#Z?GPo7YxsR@HZm3Xjz-vuYun-<;ep1bCBV'
            b'2jBo|F@mlfCOtat_oz~WSyjTkucnf~c`$xIzr>D%v4_C*W;|^mBmDVA^aw!3Xs|R&&0s8$aDe+nMHS2(dh(G~>>10y<b~o2p<iP-'
            b'c>$QN-O)9F%2j6&>301%pyh$zM8O=sHI2Y$g3HmQu`yhF8!nfmxH5Ldk^t`(v-ydR3E%anyMX8e8HOZ!ytMa*dqd@ax|O$6TPB^9'
            b'$ipBBgA5GjM3a*^+)IfnT-~W;M7FT73vjG!nije4EDV!R``9o=qoQG1s2oEm`$Gt|5wnnXihiMBxRi%B?UH;PQipnwjmpvVbaMEY'
            b'-UO4Oy1ZAdVk}21imvMUqIf!MNR2eh`++~OvEwR&e$@SiaW&c==+&!kRU04I=C60kZ*}a>?I(7|lhwSdyo#4^bE?{4VX`r&MtMh9'
            b'Us*Gc<7@l)cHx+Q#kZpHpF)c~x!#w+SG9IjYp2i(h=nHjyYSpds;y3ZRp->u!6HRTK}fe2?QKYx#MdiRO=U*JKGUk*_LI;TcR56i'
            b'az)xD@}e$X;N`MFnRgpRSRp2gOap)inwb?i7ct(ZHEM9G2ccb1ialaH(I|}OPL@ggRT$ywgVu7r|N3?F^-iPnX1CdS`=<WhfH0By'
            b'hCH$oMTPOkT&O5i_N4@aOKW9H7_0gq%D^123&WTdCP<uf8J%L$SPI~$Hip!#gsSWe*R|aEI(|ZI0qWRw<nI*;3i8bt_KS~ZspvZy'
            b'Ple=l$A$#HKMj&%$Bo(e8$c<H_`)QqBAnD%&Ox*RfbQ*g+r>l;9pxJ>gUDxJuOLcrG60fzvCtyL54OsWg&x3E;e<L|@;PS<jSW}z'
            b'<|8$A0$My(DU3x?$lAWCG>d0t`M{5A(JCaU3}Mtpg;-^FA2&J1s|)Zw^QyJJ<q{)`M_?66%Y&xdHw?<h+{Bac!Ms4(U^`3K&a%$p'
            b'Ov^G)e=*$^qmGwU;aGBPls|R}xSWV`0<j1tS9+8|qPvA$PL)nAa#F)$Du&vpqG~|6`K-Dtl|3_JPU_WHLVtWIM}aAzbF{*3*Y!`R'
            b'5?E7zuNN$Jb-6u1HB7;a^vPdExApAnsl4=VRW<Ucydk|@$eJjg2guroV7pT)l2Ncw)|ZIP<J49W&6l-rH4JSL8tel&8M`+zY`bse'
            b'tsUW!%NkFYcvl-np9ariON(c0)Zb;Za=rv_Mgakz_jbTkT&|7W3)mXOYu8-$@47gkm7cNWC=+!qo{<6zQK*!GptW3b&nqn7SAbhJ'
            b'*$16itGJ8*sAfEB$AR(G6}PC|A5S4XcE#~)?%qqe0F5&>nkI!!v+HRIh_ooz;F;lePv^0hp@zz1TRDLgeCLvgGHJ(ln+d>Y#VsGJ'
            b'^eGPxq(s|Thr$Sd9oxl4tEz3vauig$!YHkYqF!;i85B}CPmDTpM#&3Z4^>v9&1=ya<toe+CaxRjbLJvQ2dtl|4#B0bU#@wn>_v5>'
            b'J>Oco*)DJTtA)snzzsDnWszaK-eQ4P^2&%*OOaA6s6Hh6DHUC+qhV4TR0*rUsYcOkIr_%(2jmn*jtpjQRA>QCq@*`ePyv&hG{&EG'
            b'aP2Bx=J0^K&=0ybtWIQPD12l0O-Vjo7KD#QnmiGmgjg-*AXt@GM^}NgM$q=wRlU?%)tEpzVeTorHQln+q%sYjXfqi(ahgE8dV5=j'
            b'*+4VaD^-@A(&^ZDNJBqAjW$ZJxrMXpON|aT2(Q#B59KxRJM}^m3RP-L_r*L+Gvn+Rk#$*=q;W<WS89=^AnGU&pwhHdFm8n9W-5ZH'
            b'ayTTiZ%O9j*Rubh;A_z-tY6VfFPSnzxC$L@byY<=SB-bsY%AOtK-vds=4P4|@Z7wmp#f}|yHqi<QwcB8Lyb@P;FDoCeKdaig)In`'
            b'u0mD1n%EzQ!r-SCXjnFBfo`14BsGJGQc{w!0`kD}Se3+bF``(mwum(r>#gb)umBDG5u0SNo7mfBkhhEye2;OpD&lP!DqB^QrbLN+'
            b'qmftqGfDxXChW={r2x8oo~w6Rme}2>(&bOYluoQ{<MMj~8(7^C)@8$#)4}?jv>=mV6=-!8DbIDN8L6!sB3GRj*-}mY$jwuP7*-bs'
            b'gHI3C%`L9S_HcvsO-j<Bb(9bi5BB;ui9-Hy4{V{Q`FM)gH;$P5?|dh8lnYAt24nfAJ7Ij2va_*4wA9OKKq%$aURCjTL)h~b8K)wx'
            b'S5u`$gQitlUetyQ2Qar~i{|RN`Z2A)x=A9T+*rX7VDSu#DRP*78luRe=%HEhXckb#XzeJfq&>?oe{MZg^{3bGn{`0dpTV%-8y<ef'
            b'gT9mV>ZXdfWMWrdbhOZ*%a0m|<}$n;`YXz+RZ<eku;_YVEDLNi3|VQgA2}BMw(>w?q>AH7I2O<GX)d{Scu~q+8)kRdc(BouuVga}'
            b'@|P<J0I;v<t4kY$8Rrp~%I;Ps%sn!yrWNDRctg_NrVJ$%fxh4JGFJ^IQ{Wr<ZfxZ!te1ot*YOYk925;Q6Esf3kVyCity$gv>Xm7B'
            b'bC-t9x-gguc)_;oa&TGlB}Ln<GD(rc3x$@KG=9URPLqVq<U^J@Dv7W8Lw|AsC6i2(02z|ND%21li<4;n>dLG>Lk!&2lF%Nf{v?I-'
            b'1fyC-HVA-nwx+F{X>O9+WaTi_?j1!yXdFr*&|%sf{iSM3Jd{<lQu^Y`LZd2=vsI#g)8olCE3Hz&T*xu6*Ql>^nIUyP<^g&7GkC3d'
            b'tON$FE*WR|u7!&vd4sCeq0XkvHeLUM%KDfM@*GPSGpT6Pr(;H2vWhC-DOX3Sgs^Hhxh_?qV?)!<_Pf2xm6hM)04mnHt19JJyo9MM'
            b'`aM$7$C$TJ`leCZ$K8E-ckzKUo$+8!9O3RZ-?X<|Z+FB8%wg^ElXq`gJ00<%CFI*3E74uo*wu+X>YgVWp;Q-F-G@BypZ!zsaM(K%'
            b'op(Yd*O`NX=@>@uX_{ox5huNa!9^dszwW$l?Fvd=pgWbr9g)tx$*syU!c8!-v#A<R!+E4=Q5k{iWVGrvUAbl>-YcnPV{ICvQ$>*}'
            b'qtz~DWi;5u&`i=U)+DpbecBf<_rF_{KvfAfuRp<S^0d2V+zi3N@++yeqMU_h<0fiN>ri+;OjhbKy`djb)z6`-ajT^tQJ<GA<WVl>'
            b'B5iH`*T4UVqUNnFWiwJ94E@3>|7fY>9O!f4jIIhtaekdt=6$a@Aym&v*M_lrD4lRq_Bj4J4joBk@)rU4v8+j)u0dKRaio4A>D~+Z'
            b'7H2H;#`)xf2TWv=I2Ypuv)GZ6A0;*Y%<OOcE3^Hxuau^aDd+gZTVD?PpL&C#0KG()HQMi*GCheCo;0%}W+YcP#M?LGgEJ1YEJzwg'
            b'y-XZb&PxR|g>?bjs;$jGis&V2&>RC-A`aS(-R(|`*c?+2Xxw&V@AbRZo3|phez+q4{S=iy<w4iCutO=wb<@lluYCO7qyPH1{}k=L'
            b'?N*0n0m}>iu@&4Q{K+0wfLi@J2v!T_`mci^!Mkd(E%YO=Q(3l@P=n<`<3GtH0J=1eBjX-#ZQ*abrt$|)U=_3`yr=U{iSDhfmNUEu'
            b'a#=SctWC;wXH#Wtcp6UY$_k?0yi?1eYbg`eZ;#Mzo@F?>m8EGFs;tz`HD!%?X}&2bEfpg(e2twF<-%whB#HVR5dCwy9J#@Au1K&c'
            b'{b;Pg!}!Z18U9q7Duk<k*R1pd$ocfAy-x?j<6nDZ=Y$W4>y!giDs?m!?#p+rcWu@)FWW#DyL;k8{k<f<Tf93hZ#T#v1y=KVj-Dq9'
            b'W_&SPo(0^;$t}fAut+=dul^Ac{td=OOsBedogYg91bl~E6{S~&70g68w^U{!FQ;&n#LH&LSBmw4M&XwqMt7Z-H6Y%ul3c~oq*YxL'
            b'Kql4c5&#u`^QMcvTU)k;S9ks!@(>}i'
        ),
    ),
    'fx2_fx5.py': (
        '77e81ac827d6d1f820229c7d21b1c749caf18acc23c6635fb327884a0da04be1',
        (
            b'c-q}PYjfJjmf!U&`Z6^$!U1B4^CHXC5}PF6>(>IF%#_Ou0tu{ZkccFVnR{>5e%#Mn`wQ<c+4JabwH`Qjc4lj9Q!@#uTivHmpLd@='
            b'R4NtMU5;k=Em0TAjVD%7IB_Qa!cVsK)AOU=iE!@yxGf@odgDc6>a1gDA?9AV^pa>Rp3zH#LGO9@;#9<mGxr*n_4!kGD2AVo2cmm)'
            b'G!VmcF&^_Y&GEP<x@Sk?^QT`$uY33jdmIms&(8$({?PCB;B})j!gJ@j4}d1%zlDG2wS^;Q{=MhcSD_yyV&nPqo5XXikG=EL-mrgs'
            b'D3+cRh;SxC*Nw$;y-57k!uO(BBw=_f>MvjIH}+qP_t0fV69%vL_o4093t(h#;R+`ajy3U<y6<{H;!gn+t~U)`1TgK^5aKwAMYsuI'
            b'*=N09hNI&TqkeC2dD=UI6$Q=``)@*P9RR{2fKm8~w~TAT32>#bpk?T;7alHc5+*kYGr+SHtHnAd$fjYCc=t)&_2Vhv2Q~sxVPR9a'
            b'@**b*BjLoiF!C&n0G=N8S$fkOC-CE?fUTs_=U(8!&WgyJdVoXNUsw!$b6h`|TVb%+itb6j*FE~Re$+bzoF9pJyPSjz05%SVvsfT('
            b'eu85~0CBz9d?RL&2fOi`7lk5;oFHEM3GTyWn|SbA5Q>H8L;>Q!nZP3B8-KO(Txz)k#D;YjV8Ts@RRV&71W2OU0D9_)!|vsvdm@It'
            b'(~FbtuqQru2jcSVsMoJw4gfYAb~s)~JQ(a%9IpY_Rg<I|qj=>czOxuDygP3J`+>6nEihtFdLM?g4VM>U1=O^FeFQ833Wqna`x_?)'
            b'3cxqK=njUtgRQs;{i!FG4j^vhCpY0b5ex>Pkj7*Z5n%-DqJi|9t&MZ%%_7)LL}2snzUHqeT(4r`A|3)U*#aTaqmduD-hE^J2H6O~'
            b'bcE}}<|V`&o<)QLaL#-{tgRN^f>rjLRorDH7G5cCm+<bLX|mY1Ela>3n!>)XiEUMMgnci55U*ZTfyY~7PuN@jYFPmAQfzKOATTWw'
            b'$iS3H3QmCk%wi8<2p5ZR6C?Ca1c(8Fft2PXC`SNFT$gnN<O(?HiU8Sl60Re}gtI^r+-l~MV8Wz;4Ij3OAQglR>>npU#L&Ttmq3z)'
            b'<qu&*&|O~C`^O(Y4UGxLWBdMD6INMoR>xzxc3!ObQwRTU11<;yKk)jSh=V<QOJD1N4rml}dxc+`@bW*=7jvABIgb3ZUl_$4=haI+'
            b'js>j6JSE8i*X<#Dj>lUK=RWLey)HKb`u2Y59u7}_Rr1006F&@)|1HKtiUB4ag*1?q&aK*#!pp=FfX`FOS-=i79&FG8<dG8qtHV~j'
            b'ebadL_Tc6BHTY7GQ1Tj&7T%1%zSwVmPXZ&T0~~=HTdOdpsbZL7Zx2~Rzn^Xxaou9$Y~dgH@6PcTBst*Adezw5W1iaFgbUAF17AgQ'
            b'gfDBy1p$<jA_5+R_=OR-!iJd`ve%6l2LPu1tW~SGs`UNM_b4?6A-w{^8CIGge}mjdbpay+TpB`z{xFP?5d$v<z74&YBv9nV31Ezv'
            b'(n`6YEj$v)NaakwK1kBJgQ@`7h1nAn3_v;_$KGPrxD7y^z=khDs@D`hSbzXiSMmVJAD~@WKpR*j!2pw;oe#wp_){?#>XOKxtU)?k'
            b'0P_usEt5-iJ2uRnXo5PPQEu?HcMmA?0Xdr+UI5|>NKc^-Cf<Y*=rROv8pg>x>V@l}HUw@7bQmM$$IcARh*m?KoF9U`|8#zO{;_w~'
            b'J0AXuW<poIKL@Gb?f)u{dc)q~@VrmjfsKNr<0D$*r|!>aOt9+nKt0j|r1Tdv&q;uuQA@_6;)Y_mHTF|Db5LR=&k4+ne(&h`5Y3cI'
            b'!#WFt$Gk_@%{jo;uMoPz7!d~Qz@N=P{RWBkWewzk#IH0zP^X97CvXOz=~bFqfZ`B&Eg`hem(IPkNkHwdd{EyTKlZf7l$)=(3wa~<'
            b'^p2bG0zm5kjDlqdrbUdK;|G=_rXCO^m{FjK(Uc%f^Yzid9AN~|K+qF)rRy*2O&*Y#-ifq2rHw~Zgh;`UV-IN~m;!Qu0+Gtb<B=YJ'
            b'1lnR5qR9fvh7>pO2qWp}crZLZI~<DX!l6-|Whk|8DlVijP2sh0T_|P4<!X(rgGOECfk-43rxpkp1p)?9i}V}L!vF?JLTQqEE(lRT'
            b'3T>f<rVr5E3$|9kEEdlN8pZ=Kp1|lG47(qD!X_b$fWoGd$xLGb4~xm#hb?t|m!!uHSO+MEPW^i%T>x|9gM0z=Vma|h=o}}ossQ_f'
            b'=?hyl@tv3%9fb8A7!R>T?R{JLWJFqkiGXw}klqp(7NcC<Z@hgC`we8XMA<$^(W7WGiPi~e80hp8pF|$oS61Y)jg7Feg{mNf5mt^L'
            b'Aurd&#6!e;Qsi+DQ^_Yl8E9>4;T(hYx|I^<qWiA^N4$}*DE?rqk^WCbCaltnevHNW=QHd4a(Hn$#7!H}8$fhoK$ffM@xqaip&106'
            b'<2oxc<(AHhcfLa6h{gm8SXgqKJ-31z&#fSV*I;aHVV6MVfH@12!5T`<2z)>j5OGJ|I+hzD?LL@k3UU_lAys_<J&F0?f0FK~N5C}M'
            b'qt{^DarrFkqv&8fhGhbzzj^3lwEoSm-#%}FEd!lF<A5Giwl#nT9u9gZ7V=UP?2IoXz|(kaupn?YN<!oU5a=<0y>MI*-Zi1^5CVy%'
            b'l%K#*AVsHgNfQ+Mi*U}m0DU6BiO-fq0<2d{EfmIEKpe4gkav-11U3a>w4^t~3A}{P0c`|mza;XK=?$!{gR|7pcLUfzqhRDMLRgRl'
            b'rST2p;ZqG!L*j7C`WbgcO4dR5v?u;dj_qB1yR8qSa0YVljs*+qKtIe-HEd%Ih6h!cY6fRIU4tu0niq5+%)mQ8@ero*SfV?lQOD!e'
            b'2n0B6f<-J$ECsd}dtPMZ;4<xSf^G&3eB%>1UKPX%wU8&PsZxBYN-V>8+OKs{e)9sz;RO~=!2uM&>r>Dfi><Wc7ycyTWk|_@^t=f{'
            b'Ub5x{4+BiA#P?ElwI%(b9MTBfLiEC?NPj5%B-u_pFuuS8J3T%d9fIN>!u&*8rW0;ju+7FkPa&JPI6W{qJV$kxalqdx<>W;(kIY$P'
            b'B?!O`U{m2wcwiEe@FoNTI>3*zm_EtZlCBZ&d`J_*>l=_zWS@4C1QuxF;qq7rp)C_@4+Xtm1Q2k9V8W)>;a@OOLDcwKSi9alLVpn_'
            b'^8+BDiHACy{B6)WAPzm8{>EFZU=t1+s5pkG;?NiZO)~8LGDMwsc>Z&*|ErWcB>q%n0Eoy%DNh1ugMYC?3EcV|td7`oS-@_&vEM!W'
            b'Gx*Olkl84)!LkILfqjnRy&kTR9?%>BW(0>iNLa;4XB(D^eu9IG1I>kK204=WtQDC!T+afvGY6hQ!ROtPZ~}$o!B;j72#h(oyGR%l'
            b'G@RY70|YK3IZ{}YeQ%K=A;=UVk+MbHA=WswfN`n-_5`d5aE(cl5X1&ttn(R;z(RX#1qxYv@L+xiNOcYn!&I5O`gI;{gOZPB2>=pd'
            b'ltW5<C$>~%1#PIg46MK~=Yiz6!LWaMIK1rl#54-YWRYN^-gOq+*pH==Vd>zE?a&U+)*N-=kz*FN=tY1{Gxf;#ufD@*3N4S~Dr8ZG'
            b'KE*|>Y>?&7S*+REOk*`9`6VYXg<}V#ZKXD7Lf}Q$0~!hH<f{0=45c_D2nPc*R}qQJUJyuYki+P7Ec_LFKE$ZxJCNS6q`(cemm_js'
            b'PAsMPYxT095CQr!=7t<fJ<#|^iE`E(3;^!S0Ha^?RJG_K+lZp^7*qy^RmNkG<NN`hwDcqV1X64jZUBf7eeu0L{1m7lj~^pg;9uOZ'
            b'Zv$JHpvXs3-Z+cf*o^Ig0`CrnWPyV0Cj$Y9P>1T6dpVXTRaQz#N>A5}XXq{l^P0t+$hr=27a^m-NaOJ{(EWR$KOzi{#$TcY)sKaS'
            b'^LZZ~8<-j$8{~d=Bce59(I8qQ0t+_6MleESTCP<R2_@-oN^2r4cNBYb65H-*g`R|zilBNy=Ts^c8DbiZW^44(MkC=bS73Y#>PHBP'
            b'ElWNN*2@*qNU)OcHDFUmuyyG1O-CdaM-n4QCj6RB(VElaUwZwKl&FImJ^S3lc+m*)(i@$h_4w%r3@u+^Wcfsne2N2eBYphfEd9ln'
            b'AF$+t)ooN4+K3U@f3t)iNW*4?Ade#K4(PF}atUDMjnJ9c1^8%i*}v!?4|=2S;E)GYKJn<J_jB(=F6Qj=baZ&q1$8AKoF5$xMxS7X'
            b'#@Cbc&;0G;Qx_HvN=7dFqJRE=)Vq+c_Ak#eV_%NmAD_w3A2A>|Kz@Au8a_bVp>Bd59I$nqd5*(bqY>DKKn5LgMQ>J&fPe(4!f;du'
            b'pN=msdPlgYAC5ojrx&0<yC(<v&VWC)`RZ9_cKz_ab&8%q_uh|bvsJ_FzBQWB#*bn!{$lgFqC8f}&~j~ApNP0%Aurw>Sf7uNhM(Z`'
            b'%Vx{6z5@kU*O3#nfTh2xLnt=JaO;o!V_6(4ADw^rFz5{jZIP^33-5~m_R-vczpiVd(P&&_tkbSHYhoV^NgIr>dJ{i1)rUHNsv>jR'
            b'?tKmDyK4IVGK8MGVrmJwXiE$d=N!Y}+*9iYBZSP-l%k8pn0GeRXnifzNWsW7#MgS64L%Ap_}E=6EDGv5(hNsmDqPRm${&xj0XV51'
            b'DWs2P0(wzkaLi`@)c1nv)*{ukB&&{2B9us`@E{6$mUB6}gnf904dd}XIXtMjb3qFX-Mtebx(pQYL8tDSuy4W*fJVXhY}CUlL_J{0'
            b'DB7Cg3)076{Zvuc#kf-rS^S{qUp5HNBEKQoJ`q$pQd^Vz4^Zlmjno^~lR`IOz&Yee1B?e0Cu*`+!=Wong|1L{|I`sJ)hH*cvwG6;'
            b'C|ab!f@$Q2jeNV`dOYi;EQs<2WhCH-g(YI_J(`UwULr#lXt#jb2^OiTW6>ihb^5K0)6d?z;p;0;pS*lfS9NmNUFDSnx6Vi-lYee0'
            b'lh;Xo=8Vsd2JP$p?-^u1Z1J4skEGf8=#|lD3e&bgy1Ig{HDO4caPk-6foqgEe`Prt&SsT1i5xUOs#1D>0mls=EBpapJIjgdh*?|A'
            b'uDJPHb)GKbk$S;UcVO)7cpYA-TpP{6KZQXjpsdxPmO8-6X;^ax-6u1XqOYCP@q6Y<N+-@P>uF%v#)|WAKFb1GeKeF5i2WLzB|_uz'
            b'pq9K$5Yfz!0QK+4YM;q-tmA1Udj%ngznMH4G((9bL4X#4n=pnAB8Sc)bUYa2RDNtmI@%lpQg0DwkOULUb&Ll;z{l=->SbIPBP1Y2'
            b'j_b`t-iEM2)cllZpzAhcA0@6H@JAI322~a=()0iourgJ#HY3)-BAni$D#iiPuayBGw9kSVx)(yVYwbEo`ABivKMymJG7xxV*M`z*'
            b'1|L4zcpk}|IE^ODP$vdKC6p|+7Z{NtUx{TS4JPFYC{a58#FkB{+zyoecvzr(QcQ-xU=p95$P*=32iIvp_V`i1b9|Hnnlc)uf^8u1'
            b'8VcF;@8i7+a~m1I^bzD8Zl|5y<tq0;5|DP%!3gZ375H|ZJ*&1ghk&E7j)Dw(<P8U60W}J<$ECp>YhsR)`4sd24)c9QxTKS_@ObP>'
            b'A0Y@HTB)2&Li#&Q_7zOzXI2O(U+)bXnw_VD@HI^57D$skk{(b>gpsrO8k8*_hfe#N43+t)wG)a*p^q4(v0h;?G1V@A)t`tt3p`U|'
            b'0OTf9NT?MVu3FnMy2h)i=*ksdak^5U-279RCfg+sRhD&%3occ;YR%b~cMPp|;_Is|SMOOA6c?PG<}s@+aL~8HR<xTgbEZfZ$il=A'
            b'LtEgX;`X%U|9LDG`ETdo3Kp;!;-vNR)i;slu5}k?DRzM=pF*Nf0P`W*-9<*<1m@q_Mzjp!Bo($c$FiZDQiCk((d@9!Hf08=Hq$~p'
            b'%<sPdYbm%jkL$n`avRF{&<8rwhRB92tC>Du6Pc;$jtbz-w{FUtBEG-bh<{TU>X`^ujin#>%k|RMx=Q@?lXy`Rud5GJAFfbS@naWR'
            b'WROR%@DX;=AFx6#nVz<pMFIE?+goVEKTmWrzd{#<DXN7he=J(bh9R^zUwso^58(git3Mo%2U#oZ0VC`oOg3LV0hO(qXuf<pF8`;5'
            b'dhmM^Yw6Q(LI2Ky3fuCCo@LurzGb{^MRZs@C-}d`SWgzv<@E4WvHky<7;?DSO$mmp^}kCDC03U*Tfp$*&{^S_&E9eBTHO^5qZ}BW'
            b'oVDvDO~s;g7LN>ueddr}9DB<4>--qbW<QR{oMjXgqE`Y<J4#hDvu}9SMobW|$9P}Xe#G(l^623E<6-^7dA|<-8rgX&6ei=2GWCWs'
            b'r8!Io_(~j3vN8f&Y+(l8#<=yvIcO3#Oni)~1K6XOSUNYdkd4d*Jd}wjGm#Q{`DhLnuyD3%luQQ_MkAn%WHd_I3e(q9Dv&n@(010('
            b'_R^0A=G%DC07fqz5oCK%lFx{3>$aF+=(8g}I1AGe*xS3^IMF<o0bw-Y@q{4_YEUfB_t35;d*0^$ja9g^E2^<l6Lv)=ZI1qeWLT+H'
            b't0|*o`~(;YK+cSlX@|io0l-jEkw5(Noysi2^O!qlK?zix2MWVbwiRZTV`_aOj)w!8K2oRKOBD8v)7(Ebq#M$yBc54i_y8RaAT4>V'
            b'K~Cz&X}$;N={OutrcBRtNDR16r`{^jWTJ*ufm}Q~ML6*e4sgp1Xb5mY`I<vX#XOj^fTn=&Pu1&UxlZ`NF&7_Ji8Av=IKiuSoN4Vu'
            b'xN9=n%;Rms^}<EB>6V=+W(-Ni?4qf|FJQc>7sp0!E2rPMjg-V>$hH`l*If*!tjt~1{rnkigd(SQMmR<eo{IjA`B%Lt3M0D$oDGYh'
            b'D<htaC7Yt7ViH~)BgQ)7im&OQXi8}s=cSo_>DykLAjy}%0?n+l4(QZ71z-GCzWVn{H9Mfm26A47jz>?E5urGhG3cKnoHE^_^$L_j'
            b'wV}lg$(5Rhv{U5TS{BL<vtmsP3UhS4abw7bD<i+h#_OB#7Gd=Yzkv3q4R9z0PJ?ir9WFEZh&p$&5M}!ZIijlZWBHUW?{zeauQGkF'
            b'^CPjO9D%CB)szkos-B+HfzD{Mp58)dgTqc)+wJ3-i_FKQEM4X_I^$Ow5qj*m)#*0fv3cM7h>>T<lqZ?gS4<|;Nfj!Elg`obvz!F)'
            b'j|vBGZqa#+V+=<0Q=f;SEBNv+=LF9j9|e&bP-fj?UL}g9EC?z4BSm4(3hX;29M1%xi}nkYQI@KN2RcXWLocL&8F8CQJGq@;#Ni%*'
            b'35D5|v27Q(2$NU>J2OrdsMk%lE3X4D!o;r*Quf1{RU%7GWpe_~0?-|pzG=(#RI_|&0(8}`=^?HeI-10$q&1tz+$$u;nQ9EMWaw@s'
            b'0x~H9j}-%P?4?4$7?edo27aSev6S&BV>875Z=T{I{4P1jgz(CE_qsTdX?4JxXm;57=SBG1W^$nT8yO*ZmXg7SDPJn^5{RW;Fjp0-'
            b'v61NyhqQ`D{CDC!z#G0RXa<%5fywEP;;?seqEErK30$LlH^yb#g@5a1xQzMx$5=6H%o2|?P&Jcr8oi&Ooq@O8>6n&eh2QWRISGKX'
            b'8y7QF3X2oh{dmX45i~@MVT=I>fDxB3OXtNKCN|-nKx{EsjS^(K>B6|l7?Q3j!E#w3s{A&}eV4Bw#d?@g9ZWqw4u6-$D_%e_MhrqF'
            b'C;99=QG*L)Ux``DTnF(*N6T#7fK3%nDI42cEaD6rDK{8zLSxc9ACyOwSP!O^i>GdqW(jb8!^EM~9X=3m6scX)IS7N<da<yJ(xa%N'
            b'wBRU+ioE<X=+)wNlqWyxLKAksDV~ddOJCgFm5^olB0NTVfiNbi3RVI*fo%j9-S^1Zl6SyC{q$kY=y071FYtDb)WunH?MT&dqMEO6'
            b'!4vkCx9JwL*BYtZ?2{*q3GTMcf2*ZwZ<yRgL&1>ojk7{`@#+8oyJo^QW=#cgoVz}3CN5It4W0kUmqY_~U!QMvQp#m|@=JHy_|7m;'
            b'Dqb1#D5JusEJiS~VU1f!RO=v3*n{0PG9LAKiaKMy9!mFH=_<c*lUN$+ZaBpo>cPBW;$DBgN7Eb7E4~+$MyngPs^R(bT#HIDG}SMv'
            b'+H9h1mFc5_^7R3=n=bVmhC|(v)!^|WOu*MvxwRPuo*OW$NCQN~y?7?v`|NYX1J-A8x0&ZM!v{1Fy!lf{>{rDP^!c9{Tl{rvG;N`Y'
            b')0VRVqBPHwFn~8YpmR-tq?cLFb<xDg7^gWuOFP|DCw`25GQ(=bULj7<9FZXZAkv%gS9<bn7)x%keif#q1V%eIs?@5}b%ZXkF<Jw>'
            b'FW4blZ#X4B=W`i~tjZky&zSb>Pl1kaLN}E|I^Ic6yPYRUQxq9PW*<)+G*0jZ^LZNFNS|e76{qL28+s!%mDNR!tp_AW76c~|e`gy>'
            b'!eH#CmzzNlHB9EC5GA{xwyoK!V*q)@cS+jUNJFVWYsBlNjiqWj`vxlo+mXpm`f~$X3cckPq&sWAT{DA|8p>y)St~?~k?=6#UF~?5'
            b'ws)wh9UE-iOP726_^o|t_l~G&!PKbL@RS^BL(L2`g#@GuZOg1_ybGBI6LXjPd9V3moF2~b`q-=!xrzbr(zI(Hoiv6nAdr>^P$Q~X'
            b'y3Swc%i(nek8yJ^mztawM92h33Y7#7(cEGSPT8zLqV*fh&R2JM5jL!lql2y_o~g=@;`r>9Xwv?7bgu7bVn`;pP-<b2WE6#YFGmgD'
            b'ART|zy#9)wK<yA=2Be*RasN7}8`g22qgeW7I^n3=Yvf@K;Bq4x%ekVLuEpOvY93G)Ir-b+k<%r(il94c7*iq?gl`GcE9O=s)yj?q'
            b'NvUY|<Xa?kBO@1<Hk0kMtg3HD<Q}UVBjrLky#>|I1$d+A0N?;>F@mlfCOtat_vqHMc|+X1d!kZTc`$xIy^xB7v4_C5BOGlYBmC*5'
            b'(f~lkXs|R&&0s8$u!s9Z7o^S1U-FSv>={crq(z@{p<iP-d6|!{(9o4#%CKb+>89+&qve6$M8O=s0g1pTf=f%pp)p*1>#tU%xDs~7'
            b'A`kBgviXUQ3E$hLdp+m_8HOZ!ytMa*dqbr>I;A(w8YZ0-$ipBBgA5GjM3a*^+=+=QY~86~L^iNU18}Tunii?<%ng%H`_M3XqoS6v'
            b'P&tNB^oI~?BW5A(6#YWZa48OL+9l~Yqz?5UJC&p9>7?+loGB(lb$G8D`B;ut6dl#``~2ywAvNM8?FatA&W`H<`ce1qjcbnCfu29_'
            b'l(q43ZTfnv_*N_1x%tS>c(R&yl^2N8ZH`qNEKD}0)F|)U>5DezaeSqXZ|9Eb*L;@)|H-w;lk2twzO1#ST3fkRKrA%D-?`^TQf+zS'
            b't2(EK4jEFE6ohn}p}igHlK6UQs;SI~Xfv(mPpNf?V+7u>%Z6?aj|c(0PJrZEB<`_Qe$4d%N##$e!xbO$G*HsosyCk}sN<h}NI5qa'
            b'g<vXeQ)1)~3etfe6d`3uQ0bA#in%~dvI*-P;L??Zo_IS>>1Qkv70gGV5~SFr>Dq=ieaJ>W_3zC)FEv&&x(G<-Ai`9u>Dj+&s~H7J'
            b'r2N9#DWmkUO~9qHE+vo*!K55d$LG46&voG_WSf#2-U>Hl9#w|{$2px<ccuC*hQqC1eJ=FJXL1yn0y+n4+;)ADn@XD0_4jJdOi}j{'
            b'(<74@OlX&kb+p;fKcC5aikb2Z4%KO*mvcwP`CzCl;sk~Z<zX0=m*hp+z&uN8<k5Up`BFi@p3smzeUO1m6T?~CwYaq-j5lvFw9dO)'
            b'Gy2qcj%;k=2^;l?Y*xybV9F^V;Pa~k(37jJaqtXVgLv(jtNv352h5yJuh_+=&iMl(U?GYVF%YztEADxXb%=6stETKJ4;EeQ;6JJv'
            b'&lz!GJaWP<D)z?^tiyf?eocKFDHot|u13?OuxoZbEdded<r+LQe2&>7bdnb>!*ehQB<DC3M3hN8S)YLbd{WTnp^{}XYAPk#&a1<Q'
            b'?|+=Fl2%qWOx_@-+r~zXFckHQ%T141rQI6^ag1Bay0DuptCtqlF>V-}`5RmS$K|vwkPcV~Oq~EqN2gf4OI1=+mk`sfrOSx&a;92{'
            b'Oo!T0tV>o)%odKy(Mnq4iHbte`7#x{LL;rDOL-_jYJ)Oi^}A{m&6cAtEPp^wk=txAbFD-Ra26T85rYbt-ozo+r?#`@M$F*>ci;+k'
            b'Dp+R0IG+8&wyTnS`X)EWgzG#Jg@H3I#Rx!|NlW)2v_{bO)^)YeSzQZ8IbrT8`-+{S)ufJDA89ig2|=1bJ9>K?hS@+fR!db@HlHmc'
            b'!se0hrblr`4H-LkP<N%#!3N>A3USk+F27UFoxoGcnYb+$ew-KwUx=)$yd(`1N;gr9EJTn-Y4nJurOG!WtPoHVL6XBEk$s608^0EF'
            b'PZfL(I)ZXbdg&$8aX6RLp{*_}tz)b4uIf#N8w1GfDT29~W+@*#ZK-JhYvwMMjBJ&{OY~6T;|=&^m`xvyBKB-SsIz)1H&Vy`I21;H'
            b'v_Qj37oZ`gH!_XRjG#zKNyZAu1GBi3mRL51Q;KDyuvk>JQN98epq@KolMHqfd)o~1o>5EaA+A={NHd1YUKyn+Q6k@{rKO9EnntJz'
            b'JF-V1dMKZ#>Rr}Xw0Ej>S?j(KuFo2mUlZ8C>V~i`8m62J)8`XgnF**stEw<zszc2%Rn-u=Dpa18YN}fQV}ux%Nd$vW57bF5u7~z;'
            b'hxJWL(x7#e5E2jB_3sjeT%{aa=xI8h;`N;)rv5u$h#RGX(!Rk!rs+-?-=yrU?GP>1Vh#jKd9_z%{9O~-`SOfYlGCWEGDCx=Ra#us'
            b'j%;g~J7bHc>bd+et-riUBB9t=!VqBb4D<Oxn0n`<$chthl_^p&_o(!xb`({{QpPWTYGF6^r`PYB1<ll-!LZ*O9)7}8j?;_srjoa0'
            b'R!dqvE!Uvys2L}QGPdWtYf33pQWD89?|J}Jp>2jo<{In=&P2Yf9grBY;&>8{`9n;aOKu(B5;50?N84FEWYLnYWH$`51UCTy_7#1#'
            b'X=5<mHsG3{oziTf2S(MjVw{-m$T-@So`E9JbsJ7%tHEUcZ!O)8y%dGjf-vKP#Qtwy-XJqU;}d9!grCrw)$Pxpn`Sq6smUW}gQ<WQ'
            b'vUXhxE=#^BZ@ZOdXmNNU*Yb?UubI?olCYV1#WF`_Fx9{5Pb#2fhF#<#LlRiI3)EzB63t&<oAo$|fje3fX2+>MO5r@hs8*2;0-%_x'
            b'metKPH_2_Xau{m+jv^p54xOLsFl~zdLIDgODpO`m`r^t$qrzPCb)<@uKf26Vrsg5n-bzbs>ANiE6}az&ym#~?c&!*y2ZL5u`7wOg'
            b'To!%Wpb8bJz`WU}FaKjf<4l7L;_FHdC2jg>z-WusLFqfi(h#K(mdz&DrOFfRXgb(``>J$h#rHUXisgyQsxRelpy?t^50o=8=FQdp'
            b's1^3{Wm{g{y5}^{#JTmv5$<mNb#uS*=0Lp11il_WdHcF?&=T((LcZO~WckXXjFhs$Sq|!w8ycZERM%Br2D|8=|4Z+1*gF!fw?btW'
            b'nS+7p=m+m;nrPY*r@ijrvJc%~v|cn`3Ocu?3uVIt5igwStxB-KO)#;us}xAhdZ1!U5rOJhsO&XewqqyWOQ~gNVUBFowCqi=_E#@S'
            b't72>BeGGU_9=}zLRCOFIy(3r2%RXo}uEtfgUW4buXssU88~PCyPaCRIfm$LFML5w?*0R9lh`qgk{KtPP3ftRLb{rkd)6b{zFQQpi'
            b'R}x1=q=`a1z27DiQMa6-rst%4j97$?em;xRS+Vp!9TOz7+%P|ur5)ofNU$gj)UPjW9G|b1g))(qPa}B1NM@~aO+zqMEh&vrdeU#$'
            b'{mMUK*FXPEd9#=Ui@)k|((Qlj4Tb`A1!a#n-_~Wm1ZQAq213k8POFJGuf=<7;w6a})r_)ZIH;VL>W+!bjZlLZ)h{kNQS8+xz$u7>'
            b'X6@yEt3j-XXQ4E1v-ax6+s5lRBF<FBPK&dvgboL~zJVQzIo28{)@1GCk1G7*-~UTAU+p(q%%9A|T*)1G2!9Vi)$vrN$Fb-dm(d$|'
            b'5ni5wZJ}R~o5|{Lgc_`}7UE^RSlwd9k#Udr_V9<}Vp%;MRzYjRivw?!INjT8Si=pF%eEe1i8n6a7b^q6(QsyR=n?JaB|i=)OF5+q'
            b'o744zl|Q|e75)^eGP!13N@H>2bW>tlDn?8AUM*$r_`%GJB2{Eu|5hJIQ?M!?5^PMrNN4acmNifCcUV+CLG`04g<n@pr$6g`><*8A'
            b'?vd@`-y^PL4i>3gwpiHD-ZtJgS@u6`0$sd(CEi!xN#eW3%bYUZTK?jon%A-PJb^doo0>8{zX_vTie-Q*qY}5eLOK7y(=w!!R=i(~'
            b'cauH7RH`c8s+-<s=8avr_nqGFK8nIs-RC=w`j8=4R(f>TYGek)>mHJ;Sei7-FDc6`AAKje#FyQ5({OJuYjstv{|3t98qx'
        ),
    ),
    'fx2_gb1_joint.py': (
        '06cc74279e485e2d73558b2ea5ec9a5c68606e685231c10bbf5ef1bac5c2f296',
        (
            b'c-q~4+j84Tmf$<SBAXS}1<E8X(z0yR!-|n5+SasnS&}Mug+fb$1W2Jp0yF@MV)t~!Jnr+xe&Kw{o=avT6Ch>D+dVO{jw+i(CNfW+'
            b'Joh|#SgBNee=(dsv_xHGw}DtD@z@)O^Dx`g&n`~7UxfD%rfrdgliMH>6K|Dzb1@6zMUW*M@q%9J_q#8TF3&`od9$G5IA1>>4aDH{'
            b'X<r<job<)uLX1W{O>;D=iKFuq@%8iXqI-1w8GD=#PA|>{^#0iEcHwoSGs1Hp_>X`l;J<lzAGC!hrr|^2*Oze^WnvwKv)e52olo71'
            b'v+kgGdMp-!7m0W(V&6~2Vl~gg<va|MRAg~{C+e>c_Z$0f#0Tgyr3s_M{e5V=3nCafnES%Zgy)RItRDJ7l!X((gda>|9|6p}HH0|L'
            b'QW38sSoV4M&x7IV$6>GAzdGxlz=|Sof&JI9vx)#=3BV}AELfy9;YGMoSkNN&SMvauHjcAfgc;yjh~<2h5@eG&%7TZi?uY3F@B<rx'
            b'sBo|;UIvMm#fk9JI~aKyCjd{E`YeLUtrvyqLcmti=(8XSU}r@VOaj0m>@O?^zIlEa&73%zZ^Y4;Uiawa_xefq7;t_f(#>KV&jHvp'
            b'7T$c0u!R|pl>o%`X7jC>CIRfmUxOqTS>i?MBFu0f#+xjF*P>X=122gX2i_PKncjxWW#CiGJs>u&`v4PeI;;{96lFjX%?8j@R~#Q*'
            b'^^d-YLHF$P%h8}Kz8>|()%i)cSHJ25Y%c6@x=MI3*sC;M0j{eiNi~M)(#t|`KAZ>l!5sDjX8~GZ#D3|19MCpgU5X`8(;W5@umC6='
            b'+`{g!y%Z<_-w2X>80H?f;x-N^fmnEexOJG_#;Z&)7=S_=lSxFx39O3-(rb3s-hD7lU^5Yc&3F5nzmj;hOofklh{SjUghY>qVdMu7'
            b'jn!LZBLveEeh8bF5px6%5emRL4FR#PT67Cm*=tsDmyuX_rMO)pc<?6KeA9Lu0e@%;_n{^>RnZadgZQI3d{qS=Z;3tOZuqO?0Kf~e'
            b'z6F86v`8QWQz9vN5&knx1Arl(&*OE9(0d6W1_TCDnwOy*0Vr`@&MlBD;N(a|$gbmfl^`a(Ig;Q;GnWJtCIxJSuvG-9AY@?wGy@`r'
            b'4o<oNk|ZpDj1z+H>ayND{q%WYOfVX`4=<Xq%6hXp8p*ZuV#Oai_;(v{K^XWWufK^n*mF1ZwGQZjMlo}j_@xOi{|kLF$LW~k$Upm~'
            b'QOt1;U+ZxkU@hh;Ne;N~0NHai+GsfUVOQ&Qxe?H}`{$$M!I$5aeDK38j3eZKhw+eOfJsLo4J4&^r?#Z@GI0dp^GtFUumg<;8#D)b'
            b'<VC>huods#HV)q%y#BccU+M`;UgOa`nDW<G`^}$8U_^C*Bl1&c8K*Q=3RCRuA#3RO^9>`eo3FhM`~&~pd*PfU2YgvA8+&`qQ|sG!'
            b'9ylxDt3;0QZRPnOfO1kqz*7*vFycnIFcU-ert#_kz;s`<YV}r?zQ6q$rA9xdS3o$!N;Bkdko%}EU_^jRLx|8H#|biG6r{knagdS('
            b'N`f>4j1f~hITy4=Kmr-5oar|NNjmdT6#%<1dxC-iNJpbIm`@vb5vUW`@C8Wqn&Jls5Mb&`9sv0Rv<nMp1B)aWV6yXzf!F|lD&|66'
            b'l7!<GNM{FNzD2QRa;a{|hMAX)QO7gN4ZaQ@07W4nXMM{HKwJUoDb&Hl>o@^jhTu)&G<#3Ia6Qz9z%76dQ>6UVo1z)fYKSiv#~|-N'
            b'Uz}Zh>YjH`2fw44a3nrlfYd+g{Vq<rgYNO*qDR_+je?WY6I$cvqu<b&VAU6ZdZYzN=`W^%mjOMamP|#(kHunR?5FT%pu|X?6PTC1'
            b'?#byfnkkira~?-ed5^4{3xKOvA#}wlA`H|)IGuv}jWXxk3djSAUuk}z&H%Yj;0z(tt2DI$#WC_)MrfZcya#EMfZAV%puX2(8fc9v'
            b'H(zlV@<tly9XH_xfL0M01&bI=ixf8}j2ur)0w72*qd*g*DM6a%`=fz5!U&*&peO1|KU~zCJRmW>muPiL8;_<4kwTcJ0n$b^0ptJ$'
            b'B9)CsLp}Zww8bJulLeFwDQ@5qM$*Y?e{gz!JP?z)N27R)SZd!~Tu5P>!|UL>P|C)O<qBB`jk+WNkw_{|Ef6pY1Pr1U={KIm5e$;W'
            b'(j*Oh5Tbw-+Cm3SAE3EcY^{J<EM5vUjQe0bfzjI^9DV8vmxL?=3Y$tMGmQm2EXJ!4w$u-Ok{-8U9iSLG3m=el0nBj-@&(L`#W*0L'
            b'bDF`bBJ2yMFKp2`^ipPY5Z3o#Jfsq}4{Z^W5$OOXBGRcqdJ9}wigI<o@$L=mH;~N&W%~?8kD|#eS!JYQpwnx7k_2d9IZ41aHp0dh'
            b's)7tgSbAZCyj&CG01+QZk;gquC7%IhptY%ma|+h$MoO5=qyGVLq-*(#;t$3e>Hl10!YYI0msDJQJ$Ejy2A5X@+_XNu0YoSIWVwni'
            b'FB}OOnnA#MzPBV(Zs9F?=PM+RXiT7hxg)nZ@GH3S+zJwS1;)k(b_rAtn6n@moPpGgzy~w|5qA=-Qn?Y*?t_^oAZHODQq@P$lb8?w'
            b'C+UuQ1Wc1XdJVQ6m(Q|3Ne)IMSSCRF*8p9N*8g<tcQ0FD%Rpz)IG_iWZ4ID-hlAdUgS^xPJLB6B@H843EC`&9k`TE71bPZ!&pjW6'
            b'cTH$Jgg{~`6=pCLNYRO0(inyQJf5*GK%WS360#+c0qfOr3x)9(5l5^&<Xz+$flX1IEa=U63@@Q`KpO$tFH3@Katmwg;4F3Y-2nEl'
            b'C>VK*5EdjsX?(+Y_*6sGkT{&Me#TvqlC^(y))jvu$M!zG+tde1JOw#;&w>SYU>I9e4cl0Q;XxJVn!%e)R^UpK<^>%HGw{xj1B7Wb'
            b'lIXTH>S(kaf&hn2aEOJ8rNGu=&x?#4T&5e3(anH?Z$knnsDe157V=~@Rf=y_iDej1`?WsGZ(aa7yuhM4IDi6peF8dTzL8e^JRB#y'
            b'3@I6qp4TzROV*s=VSs6sg+Z>aHl#n4Lz;kFh+g;v=?`U}B-?QS#us>CXQ$`GV^G`!n4c)ibiz#ww%OR{DP;2orw1m7=cw*12mHNK'
            b'PC+sa$ecA+f&knCHWmIv045;`Z$coT1N<n5>63gd=^F9Q$21|lz6J?J_URFlz#J_+TpkM{v}IE5p`h1`00NE>OxV;q{0k;3h?-Cf'
            b'Yd@GJ=r7`AVFUy;4p3*4zYSUk#9@Hb-v;v~Y{EeU6~_Qo92!HQNe10N4^ZbFU;NhX{VwGWi9Zz?03z~H%98+E<6o>$BELQZt0N73'
            b'7O)#`>>Zu|3H;|N$ZV9@U|9mrz&=OuUXPbZ4`>bmGlD}MWvpVPvkgl{Kf%Mrf#yOqgB;01){0CVe&B%GnE}tB;0x|aIDtY6;47O3'
            b'1jdZqT_lV#8qWU40|J+k94V~HzIVuw5M+vwNZBCn5NjM-z&KR|djeJjxW*(&2x0>+*7*!a<e<H^1cj_Ucrd>sq&g3XVWP}k{W_1f'
            b'M#;yr1OQ1e${{7bmpUr4f;QAl23BC0i%9ZYf6%)+9$fXhVv@vUvPdvd?|So18m7|7aCC6S^=JpDD~`JG$SDh3^di8fnFQqfSKnha'
            b'g_cKg6|$(}km4ea9b~!p<|{Tf^H>c@e#r?;;n)FbTd5725O~oKfJUM^xhi2aMJdh*!ok4IRYc;m7X;E8<S;rN^Ki+Y4>2nF4x~36'
            b'DR5)$<%oiplS(Q6M!g(nM1Y};xgm#A4>bN!qMUd8eSrHa!syp5S1o$THlk=W0+oSbmC*>~IDddAE&T{TffQTDYXBlfUwm&5KSe6Y'
            b'6Q&3j_!qbA+rSoPDDsh%*WUatHDi0Az>oR^vOq!hlYsz4s6%zky&TJvDl4ZXrKfAgGjtcDS<PWiWL*ciOOR1uq|xXF=>9#>9|;CW'
            b'({D+J>c>ID`J#u84NQ%W4RSxb5y^_NXb`Otfdw03Ef}FGE!Wu+2_@-oN^2r4f0za{65IZ8iJpX%ilBNy=Ts^c8Dbg^rz`Z)hC>l9'
            b'mSB7f>PHAk9Y;QkR*NOkNVJsiHDFVRuyyG1O@<^EhY}-5Cj6R9(VDZ<KX-dWDN*}1diJ%8@uDH(r8~Sh@AA`+7+Su>$nqCC@)-`y'
            b'jr8%Ow+QDOe!!9oR<~ANXd{MT|4lP~APt)#f;>#HJD|s@$|Zo2H$-P*7vRJGRqwKQ+V2jJ`o}z=@`;CEy1#Y5$i<vroeht_9D%x$'
            b'4=zql`oqt#LgVX~i?96c^79ca9F&Y)^kwhj!?1fPU+rC;TVr1hKb)S+&z~?L*GGPQ`Wik$+p%tf9qhApY(2-}oZ%2`Lm-2WxTZHN'
            b'MnFJ<RAD$O{m-YDm)#TG(~qa0^wUewpGRK~ik$&}YV*}IYj*wcp>>9yz|ljP(q^lM=|gKcrHvn^VEm=#b47WqkfG(qaXu4q!9rfW'
            b'J#fCBo(w+2=hw}a<NO2)uC5~|XaP%qRfkYqjN#Uw_{VWLRzAG=__5y|^xGm^E$6{C{~e;a0e{`pM5EET!C0qTZ`Q;<7?Lg+U-c$_'
            b'XsQo&{!~Tgbp3}K(0A4J`(+3{cg55a3elDnB+dnf!MUf_4Mqr=r71-h%Q0^|)aZOK)JVa|JjB;}Z3iEP8GJmN&m9WtdD09=Un*YB'
            b'*vcP`>;Rlpj}+2JGXcFQFgT{ua1sX5WaE%(T98#oClN{{Q+N;sJ<GXIy^E_$Vk6jsxA+W|(rnypvtK3M>gK!sI_Q!mnI|Et+qphg'
            b'VC;b2Kul<FTON2ULe!WV&o~xoRT_~FKx7ad4atwEgTd+fC*i(sHd?R62ed!^I=WQe3`2YH*8V}`6~+nb;HU!~RnhCfV+7g-f*9Ws'
            b'hX;+;KIodm*Nr!C8nn^^2f}g>ak-<>KDl;iqZGn37^HkJVswruwuSEN3*p|zYuE%jX0YQsPAR?vLng__4AhY20d`&$wR(y)>5*>('
            b'7R_shvn*MVL7xdK9UiVI&;d~D;FQ#m&a;A4V5)i)GXNM5C?eKm|BZumjtbJDmGWyxv{a*<tj>y9N8V_WhOo(FRP4^Uz1Gu(OfGCt'
            b'l1egWwSXA=V34p1&Y*<`Bn7s&X-%N-vo*ZoSm6zhOBhiN71N-21}Nq_1txqIe*n2OCEJ4{5sotMy*xT#MJLd7f3yEHgUrWDfw%aD'
            b'Y#2Udmb7fBjkZ9#x`wVbVaV2a{1@Q%8<ebnV~r3`r<FDdS2VM#QcrvXroqPwe;~oU#n|`6v@NFB+<c=tPv+@Ry<jMIFza@_4lh)0'
            b'jAr1U(x787OVpr_I;twz66OrLk2RB`ubtEJd*({EPHZphd0@E4ii>Z)uz{>T8A=K{e-F+rLgVS6Zh4s?qM4rn>fe#ozLE&8<GCyQ'
            b'KQV~EsXPNUgQg@wfEIz9Fj@~HhfZ8{<QpS$VQR*7+8pdsZxLsZ1QUx@ibq_)$Np*(SWcP|{E#BY52m7MEx90Sel0T4O`EZg5?2rS'
            b'i;8-K(vIhOB7_RfnMQ{;m{!p|p4_3)fdirkFGE{siAE`OF9oGn+DVl1ks{K68dxzw5O`$4$1*$sN;g{v0m+;+kDJS&E{1z0lq|L9'
            b'7<(eGkYytcCglm33v|$nE$di0SSb7PSVVc>nD&8TGd?AfXJ@`XMxX&Xwt<!mM}jGIE8~7DtOx?Hp^#1gKHYyZw~_HnA3@&XcDnX1'
            b'SGfn0fV2n?hG3B_!HW&tX|=661YDw35?Sm~G#rQp)F{jzR|a#eh&i^*r<ngAFyD8COFlUZkEgEm34-9UmCDH^r2l})zJrP4%nAY9'
            b'*L#a5b?3Ptd=Jxw1=1u>qz9A|apKLt2W5-Lp>xe9LuEc{?S$gV=p!!&z<fKz0PwbrgBNut^G#Gxx?#tQY8{5F@nYwIIVevh+DC?c'
            b'j9^JHG*(Lt7w1~_Z~7AnB^xoQfAiUD1+-Ob_PZ8p$_A4;kkz#~P8lPX9kAUVFp#omg-Ogm<&o_^1|)xHUH=T!mBj28o3TKj8q%_i'
            b'E+VBQRG)y8@^o7RR_345G&>PQP}$P9yx^@?np(5{@{XYu^L~G|+bu#H1?2_X(>!IhC87JHu$4{iU39x+;>V#aiQV${wB-LqER{v}'
            b'&cPKdU}UCQ>-FIek>##+mu4w<fhnIuqR#;HG1}clMn43m5;canmo}cZ;O)5PRvX@Sf_E77>uo-Zw(Mifbm_eU13S7i&$7XK_8ZDx'
            b')u)}(?6l+gRjbeUL}u>ot(4CAkxhA9#`h2VeQ!%cy%5o|u?VAZv0AuVr;1;H6|ZXIP4#i=!xd^Oe(EC24D#d^KEag!C#+CQrsr*D'
            b'SpfdP_Lkc4&oko^-=T}r6xGs`e=b_d>?O1|4}S=+2k?LM@Sl#y1KV(a#BP5Klg-0tpt4mH&DYPz<^L<89{eMTb?eg~LI2KyO55^;'
            b'p4+x-`<C&xmC@nsoZ!C~V?A3yZ>NXnitYcOiJ^dt-IQQ>=l`|Du*K?9W(ycz9BnO~LAyJ05~{lbsFWb4GZ%LqGpq2WPH~g*)2|!_'
            b'PSZfyew|<9>GYS;h*R*QQovK7X-9c{W@;L*+6W`#-Lc@?DoprDXL~^T!|9;@@uF9Ue~r{t6+M-qYMGZt*9bVW3;4=B&OkI`p=@DB'
            b'!P>YI#|fGeHp~i6sRP)fm~}lfQuK|q9z4fOC~cRr-TB-a7BKfVd0<sXSB68NjBGf}*$VG=<WwN9)1d9FpY7!z4a~RkYzmBCI#sfJ'
            b'PzKzPZR@reW3;>@K6-Q05!l<iTYJeYm1hWOz~iA*9_XR)Z5W_kP4>LY{Ts`8=~h%@r6$~p%xE6|1<9~dt5$PHvHS!W2|%{SvD#s<'
            b'N(3-eRInKT{G`(E@CfhTn^SfnCrQIFl!}N~F8FZzbkLW%Rq|vTPYTjJF;)%f$8;o*2l^Htpu+*ACGY&mNy9WxhT<ePk53>d4LTnZ'
            b'WAu|ru*@`>s9{y?AM*+jPQ0K7+%g3k0$fls^FUHD59ZCGDdNk8^}1NBGCrd&94V+Qve`Ty<2_WqhTtW*Yx01C$6Lp%xsPts9XnCX'
            b'7?O(FMN{Xcz<84&O^w8F&Q0?hIf=<AfB7g!cQKr@ZSJD(7fxv-6gjmm;h1=Mkp3GcCw7x0PTUG`HY|efQw8$)!4w@8lkn0MG1d{+'
            b'e1{E1Q%=)3FU=fE-}c4?Nxu9YXr`4_L@6OT_~LK!)xTG&_JAfE$axhyo;-~uLUAf%&_5(NWxhqLB`Ak#LyH@dD>V&ir_8lA8_Eu|'
            b'VoeJQb9B6pX2^$YBl*q6yUp+xVf7lnfcB>i@F+i2gK%RHXH7n&&K(;fyMI(5sv19*Pw8fBN2B=K>U&chi6!L_R28mf>+qoJ={cPN'
            b'4#%s>9dtH0?2NVDJ|5G_WMxWWXHKKE5Ty~J$9`KSM9>Aa58Y1~C-zKvV&#Qnrlrm}Qu)nv^opP5BzU=6czCUmj_^EVFruIOERKD_'
            b'H{&^Be(HrNh}3{GB^{HFQ7qXY<m`_Wg$Z4-@05){6@V_<FHlA{Rfzy}j@XA@NC8vgHj{P=JHd#<JpdC5Q;k#CEpHKKctx%?P8F!v'
            b'&o;}T124kFhX*<P;mj(_tEN(&0cR2D4$N(K<$9_%ADRGNw`+Qc8-|W1aVewI<uUh4S)-;J11uT3TZsTGi{z<dAWef@2pEI12(aKc'
            b'S`|wfk1U%Z4gVS_9>VXEgG>mojdyR#6PZ>AyoqLqTYO%IuWe?|ihm~~1drb=Y?$(;0xyAB+68k}p&A>R{%}aAY{dU0E+Wi@TS7Ci'
            b'1PDyd1s2EMFJE-xgf@X|bScfai9QeS0*lL-zkf*;qoyqJI6Yf48Q(DpGqf|X^aGvjldK3EL8Bl6aCYO?lFEAW(t4QgxK)LQh%t;Y'
            b'U>`8zb0vYIc*9&cyfBL`2CGp{)lt4MZZd|XJAANQ7KkeOlP-qHSCC=@ylxOp0?wd#Z{rnj=NKbKv67RVZjq_M1+uTiEM4mW@kQAP'
            b'Y}|lN6;J4@LNK4F78)sW9y1*AMhNFtBy{TpOe-G`Y9-AQ;QIQNN7-JSL(wQxyQXsx2GiAi?v|xTSw(5VQ4$qJ`DM_n!|T|d8LX?Y'
            b'xV@%$DS9n^t9e%rpW%z}82Rnml%y(H3E%{_5m<EJ6K_M_0SEQ-hc%<aO)k8^+c{CU%E`4ORl~WBp(+tU*jL`BUrKmv<Z`n|o-p42'
            b'aAnG5Ezgw1d_WoshD>k0CAy2(2LRX&6Rt69E{M~@-Fh>tm&(%U<Y2x*9jRg#eC3&rZ1ZFRy7<W#>VZ=6K3zabI3cC@f{6`l+(@EY'
            b'MR|@V?52?ptjAN-neyFfy8KObE{tpT(opx~36^h&W(^bf`tt*t-hf{5v!EPv-LO#&FJBf~R8Fd?eo@uhS>#owj|R%uN7Qb<)E^iQ'
            b'b%(9N(|MeMuc;ExEd^c}&{m`oBH}^35dMSx9Pxm~i~NoCTqcEq27)(#?TG!V_#=J(7seKU-55<9XyTlHJ3y4D1``JGx)XG+36S*i'
            b'3P4>nF*3$?0AA#s9;g#P#y-}t8nM@i6EsI8$bS@hnTTsWxgExmTdZG&DJkdGEsQF)>SUFm3v7(m0PhQS$kiK8w*rdUA!Sx&j{aB7'
            b'i4P}0$G5Sc%OM@_B&Xdi5~L}Lj3MpEV-JlJe8GI42RHI(maO88ZgxX&Wg5S_#k2{4<j8{Hyz`%2Bd;8c-TdZ12%?6`d=#Q&_tSMX'
            b'TXhT|ulb@?`vz$!7if)iwQ#YHQ)k~`rD!uW*-3wHKuf8&+=6^(&9`f2a8g70LNsfoXfYBV-Ud-SZqxP-HML`djiL>5kDtD^kL}(G'
            b'RrHt`g-o84R&S`8fmLNgs?fGfw8x9Wc`&h1@q+i7Z;k5V46l#PI#DR6@jlPg=g~=H=;|7I?F==dilyr-4_rB+uHZ4QwHC73^J*bh'
            b'aHLdb;~32?w&0Y_3M5*;#j7jof;+;7<)d`amBcfZz)~D<*9#!+e^2N7p%p_ixrMS(qs&qi=A|_?c!PBOb@S#sdIGgWf=RQk{o?*j'
            b'K{u?@B1f_GvpV6Z+H2%t4d4nR8q2w+mu|$rcho$fEDG|s!z1TQ@D)LK)G($*C<)&!Os|+*ja1uqEJ#Yl&iEVBb)`aB+Pt4(v#P#K'
            b'Q+TYdQ*9T*`4z!-A;6m?2LK09ixG6?FzLx@uS+EX%<>85#VnQW%!Bd!`E6|+j6DP{tm0_{8R5@wTt@&ZMuVkMY6fG0gah0ssvck_'
            b'i<6J6V$WFTB(I282>lwv$(ycpzl`p)Q&KjANY}{60WA;wCJN^8^<V@(6I^#Cjg8^byKuQA#g(xumIQc-md#IeO!!heU3x<w$S@?)'
            b'<E6bf+#9NM)7g3rvtiOni98IFFv!4QPBb}*!@ZQK!quHhMq~pkm;lGRrfHGu&cZPHw2uu_G%9MAg~~C6vOk1S8!-!Mr|1_7hD&*9'
            b'(=N%!A$6z+*{K{&PbY_e=}j;hs>6HLD8_QMqUfleKNnAD4XKf4c|Y(6c6MAv(2u(RY}^612YUIkv#pJfYxCDz<+oaP=jIbT<H>5?'
            b'Ro*$vw>ed9urS$}Q=`0$s_$l*$MKamzFjz`U-9)H{HM?&Pp;b%_-(B%)!Hhw0%D;F{w_Q>l4`dnzN&L-=wOkeq#&eQi}rS;OXBNW'
            b'Q%z+?M4M^VZu?2-8>$>4M!6#G5_uz)ZmM$qo6NflA}oaxMWzA31I^4*lZzOyjT*%!)q~KkoWve6o@f+Cb0^EB`7(@f^+98~X}@|^'
            b'e|1o6y?I@4y?s-CZ$OyHd_x{tiK4=IW3I>(D*IA`!8N2ZC5%-e3}s-BH-%x$QVJx_xn4}MIw}S5)5eb6^Q~DRuIrjUeic3#2y`ts'
            b'UpAIA?rirGx;NkL7t<1SOl!CdBA-2*0^{Ih03=afp+zzqw#tu%9za6jggRXFsa^wBtE+nR0g^iJE1n?~#-f;8rERK+;>l1x@S{3o'
            b'8xmALEXv+OWG-9Bb&hE2x@yn7L96d{#E9Y{*Z@+`(sXUZe0t0&Itw4ndvP_kDs)LC>*UF_qw-`A(>OQEGf9Pom86FGW0!!-i6|!!'
            b'i(qmQPx%tMTgX*1DMp`@8rH}#lpNK{0>aH_)m^EYl@Vc4uf7!e;|n<oOaYyv6>hsOYe5B`>iT=NV3(>p%lR2u3O<uh-WfV+7hlii'
            b'U1qCTjYn1M=;cC=Kr#9)OWJ{@N{Kf{9XeSyATkdR8$~o<SH4v+kV9y&kDFwq+QhK!tdzHQhzATCJWJtStr>l4JclhUp0QE?l+DWd'
            b'5^Q$`1bjX`0K;gxF)|HcYY?v;bJc(7;DA;#!jfYu)VX*f2rNW_K?Z`>a?L$&uqs;tZq;NT>tPAn4*sK>@n9JT#xq9TqH=$XxqIw*'
            b';@8}nm2v?ZXKFM}3cF_4(-IJAQLe!=!^xb^V=sHvGF(N2Knk8>Nko~nV>^ii;IpE1j}>r~VR9+ac9s_~f<32p&B$#<6lIkGsz_i|'
            b'Ohi$yxZDhZsCyPhc`&1rfi80?i;U(a#*8WiX2K4a1M@jg52OQ@MN~(=(z7j>GE~-)x~QCQEnQfa7r50zWd7TZ(uJ~itX)R0Kr4B{'
            b'J1UV!2?bR453TYoUAD)5q&C<ltp1@IMYHAb8_OS%QxqpLn7Ot^3vk*Xy^(?nnB1l@R_k}|;!Ec6fIIO0I~6QCW1Po-V`oK4K3(mB'
            b'4*}{t5ykzjmU75vn-4=5U$jQh_SQ|c)LE4dKsjOVDI4*fvel%{)t_iH85v-jKs$PS8;03HGgh~%EIWbGv2RR<et!09RMc_{C$rZY'
            b'9c&QZs51>Z#pidbg_H{_z?HVeJWMm=q!W>KU6iD8M)^f*k)@E~Fb@~fv{VOdh*d<kgbd|yNMzrV%*C(eWMc(igU%fNExq)TDgS|s'
            b'i_unZD>UY+@viGlg&PA%``F0bOtY|=o43?7fHiZMwv6m-g_r1|!sl7=$uOHf8b9g57KBR9pyEq)?2kiXs80(ttcC*`Vsb0<Ce4tU'
            b'l$2zwfIP7LGbOQHjGvW@=wa!+YGeBfSbzrpkWDh!P3-M5$a_X*wa2(x)n>K~mA!40rbLN+qm~zPGb+5HChW)_rEsx)o~w6RZPwkX'
            b'(&aCIl+I^t<MMj~8(7^C)@8$#lb?EgL@!f66=+oz7tVF48JDaYB3GS_*iudXcF9wO7#58MgHI3C$t|wO_Hc*wO-j<Bb(9bi5BB<Z'
            b'i9-I(25h0H`FM)gcaE6*?|cJtm<vkx79-cDJ7Ij2va_~Bv{cJUFDT{JUTx#=ny}|9GR~GHR!x-_4VqSIc~Lvk>tSxo7R}Z3_Q$mT'
            b'?M)I1<;E=x0T$1om=K1!mLZBPij|p#Y-Rx!0@jYAN>Q`?^5>ShRDZg?o>|^e{pk;S-NEr^Jk~n9+}^b1Et$rY*ODwW=&F*&5wQ&B'
            b'hW?6jNR^aCGAz0t7|R0N3`15L>_-m4zO6iv7^&iT5{|`FYnn@L9bOAE*M_+aHXdxW<SW??gZx1P0s!nQ`s&igV19MP6_7hy)59Ja'
            b'Rnv-bM7krzY*#)Gia_6Qc$upPlL_Cod^h%T6jn>ZjLWl!e+`NTnF$)F97rVmiq@=dfBDihySYnE<`@`E1-xL}bvd{!`I4gTwlyV>'
            b'!wZF$7c_p&q)wBB%~U&<IVxqX{zHFq0VPu^lK>f#z$)CBCySG4{^rK4wL%Qs(UQ;}r}`v?^8}+>MK%b4a&Drnn`v&6+hpZ1)b2e+'
            b'KxiB~FxO$)9Q~ytIXskAictFE%0i>!eX~`feyZWgr3kGGuw0)nFT<xVFPR~AKGXnt`U`lic!&Z9t*!)S_^yR(6M2KGOrcI<%r;%^'
            b'fr{9e4f51Rm+siorq6<mwqzA;eWzRtWGjT*W|Qku^(b~U9qhk5+`6*zdmKQ;5@y>fv=y%$>OyUg)UPq-EfjjGmG<#<Ti(=r;JjWu'
            b'h7u>ZyY)BC{l?n^@c}b7yZq$co5n#)d}s*ycFRiZ)}>Q*T8p~Xh(;)tJ5{&;E_)aM);%6{PekjTP^o0*U|>3i(R-REnY6@N_o#o>'
            b'gYK_duNtof9n{ke!oh(^=icN_Wz^s%nAq7>`=#bQQlh7fKy?nd?KNGeWhdUZQp?UtEJmj-C6$a;yKs|TkkP&|*#7QidR52Kytf0d'
            b'$z$4zaiswV%kNTFiXs!5jVqKDtq$S&AX%x$^oD*!br}b$3aJ)kL`Yt;kY|vXM6<W|pa1KBE5h8{Q|2EfaL{j1^3O^-&XGRA%;=75'
            b'6zBIKWy11?Ga2=qbl(;0JJQiCr9I>Cm(b}#CTq6yV_7ve-GJar;z<33&Dsn3`e7^+p7{`i2TWvkGnY#R3)hiC9EC3Z5a{pxJD|Ob'
            b'uasYnxzhO4NMDY6pSt~l02M>G44UuiGOva+bu`~0W+a2w#M?LGgEJ1YEJ$ibMME4^&Pyc?g>`e-s(#ErTjwQdP#*(>Ar6|g*ZZvo'
            b'aV(yh)40vr;j4FzH*ZC1{j@;-3l}Q!$AhkKVTV!<(WaR*UitXTH2?V@|3@?r_Zux{Yi51^ISkw({81HEB3S)?1J-lo3Z4BR!An-K'
            b'E%Y;-Q(0J&P=f^=<G;w$Yr5@>BjX<L?cuL)rt&8!U=_3`ykPQ9N#VV{hBH_LxoqkYR*&S$psBJOJPoIH*8<UQUIyhTv=ndZ*CFT{'
            b'$}*hX$>O03RaQ>ohSHt9G~bkzmWn|ZzJE;_USTv1l0^MdgZ{x&4#Hr;O(fWqe%90AVf<}~41df@l`2)gJXQLM+I;%+?x&-{>2F;!'
            b'S;7azb;>a&mGhVi_r<%$yC$oQ7fqmx*N5Um^}Qs%JG}HO)AHrdwyJqON6!-lGrrL)!}{wuxuXyWmeEH3)j#vVzmm6z>3|mRq2nF)'
            b'fG;1bvf-*CfSIA?mg)oKbpTG1cv%nmF04MADAeX3-nSaofOsuPaurLH#`eqXG8;_aUEksY@4At?w`Ut`Rp<W#5Jr!2'
        ),
    ),
    'fx2_gb1_pointer.py': (
        'db39d22b9d4b5590d864d9c3676accaa5e45e23b6f3649698b7353b159b2a593',
        (
            b'c-q}P+j1I7mhbfyIdt@NAp;gzl5M<nES9jX9qY1y+S8%XLO~U%Mo@*T3L$!CBKC2gH}(tXOZHqcv$8J8l6PifV;$W#RAyzKJbCVU'
            b'@=&Q%Tz5H|J+wq!B)6VeMd8Gm_zOST*3T|ZdY^^!;Kywd`O{l35>sa#I}0)Q!ljo)Tk(Qk8Vq_byO(DoPMo>du&ghix<fJibUF~-'
            b'laqlMUWoCSr)iGIHPJmk5nn$2Dtg`HPuSyhczSUzp!bJ<uLrLioe`dU&wT_m0sk%hd#^1VG4mfhx4sJfAQ2nSpWh~)Ykllpob`tN'
            b'(_^vpoIr##5xQ<Hmg`00uNJ-+#UcsAJ5fJ8+HdT?7Vn|Uj3x|@_V=Of&I@2<Z{Z3j5so$Sle+JELE=vV6RtN6T?8=g))3+}iAA^x'
            b'VA<!rKMqHyA4dJ&;OeY*0xJrfCHCKh);a)$MF6Al6K@&UgcIOOVL{8#T`xRb+9XVF5oUmADOQVhOpr~(An_iOy6eYNzz=K$qQb(a'
            b'aOFi#5=O#_?_lIv7y&#z>a+Bww@%>4O95L+qtCs-gPj$TH}wFAu)nYv_~y8NFt@^Bu@&9V{a*Lv*ZN8C7;t_f;_Y%0E&$j#6wYFS'
            b'u=xp&6#>NcX7i1hMIP+NZ(bCNByxgy=_j}klWpR`Ye6U$o)ZO#17`w@jBowb%5$mZ9uOPWU4RKU9aael3KAfRW&`M{Cyu*UgYIWB'
            b'?44bH?hbq6OLriy&QE&%`qcnnvtfthb;N_gUd8bma9uS?sxgXJPU1U@(Zaj;7O)>U3(x{1_H*yUkhbCKQmlZQ7O;<i1wi5O7IuH*'
            b'#6SV~h8Nw#F!!((x1m4v#L@x8ZT#dmTqlCT02I=gOd=wTU|lqjUbD4v?!8$An~4Z)zTMaS6@}|nEL_AxASPQNBziRR1J`?KtluCT'
            b'A()PEeb~H&n8UM(Pyo)E4~VtZqFb=aezS_ZjKsn##qAQ_gELJQ+qPv1_(N0J4>hr^ijJ@!#P7w?t19q#OY8}I%U>-E0A7mCEeHgr'
            b'MFJU^5=p@c@Sj=i0Sw_{5pH6H-iZJ)ATW^9oCM_vK#A+JZh>3@CtVRByH3J&gqUy^NP=6<ToO!}6tLmLRuQCvkb(W<1c(?qSn(1_'
            b'lCb<Cj0n1`%X<Ix<ENo9!FX&xylBEI>&@zTEZ5G96~F7?-)+DJVc_??{wCsJ&)(A4I-mm@#oS)umnOXYPxQqcr(=#I|Lhk=F~>PN'
            b')Z<vdTFg_D9B|zpvgdfb)o|{^uGZ^vBcN~ZkKN<p=U<h4aQ(y&1LS{;@sMJGNk<_KB&BnwwxsYfaRlJ=OmY^m1C0k8v;cYJ1i<RB'
            b'6>r}(j@}*|{#b)A^#~=e@o3@A`0K0v=8q&Wf;zwvxUsbgW11?4Dfae|HT3)Gh7s2-HqI9Qf&cCue?gK1zN}Y`y*=is&26~wtTpge'
            b'BuDtVc3coZDJdf0F^FFnaVu<?i6ML4cy$0^+AmtQdaFv`-~5PDV-V6SAe>>P3Gz3{eN-1PBEY2~MCgyh2pKW(V&L1*i%9}SUYr2N'
            b'h$*d<3);dXfs9nn^y`Bpoja%sfL)k9LBRl|<8kaQW{tZ5)Cp|(5~O-f@q+~jFm)vlfcyd4g$1;MMG_1!+4;p#Y=J)&bD=JY{K*=m'
            b'vjs5UqS!LIRJUWp+=(Wr;~C`!UwaRLA|H^mx#a~Qu7LCu>R{qc7=bQB@TOs$yrW*Y9%@72mOzIwQhw~r(2Qs`#OI4+koTW1&MrRo'
            b'&U>fBU(rnHiuV^F^}GFF#Yu12J04#2NjtDnaB_M=Yy8yx1&s+-eIBSsT7Z=PV&*vs&@*buSXA6lEVssf3TF;VjO00idD-usoF1c@'
            b'QfXM{Vepjq$hx@zxcU`BR~RG0KpptA8K~bNvA(W>JdpU6<_GHZkoyGA;4{5SQwvZWBCjQc_W9CzkTwaZ{gn^ud*jER)|hhh6?Y+T'
            b'#Gc-96J7vl9e`1=48gRBadZ5@a>UdFf&?=PG%=bIq-nlA8ki%D02&B-qONrPWxdG*64N`8R;RS_Xo?Ui_;KtZZ3I(54p1Oc*?2tC'
            b'<BvdFEJHL|K-rMu1|DG~otzGar{~8*F<m$`in9!*_D#iw6s9S>7Oo4WY`9#lk#*3hi#!mCq~g>90i!^`AZn3*!+99MAW0}qQqKh;'
            b'3P_<Xw9xbcntR383Yf*>r9i`Y0LBv-y@O%*V^7#5WD!u<R5F=qEZ|`=S^Kc1uJ4lcxCQF~#n74mfTRmxPJECrU|uXI9toY(1XdMb'
            b'Uod@PizdDkGoypBz6awWmZ*Jb3!jWg3osFoP6g6i;=*E-tNV?&uVKG|Y?dh7=O}s<O(xMgAq@ka4)ICkp?zgV9^2Rm8(XLfG8kdy'
            b'_!07QO-wvQyeCB-_b`=w0+fN)rWVdISg%_tVJ^G>3UI_5`HJEX#v1AWRAj;`z38V{TzomVF0O``S3}&i0lfi4CkAA>iXJZ<2^pF}'
            b'z&Wn7B2#YZta#@uB#vlIpn!!Xx7l+mxbfTy5_k>9#uj!7R1TQ4AQ`No)QrFfGyxHJ<gH`55z_91nWi9T5g$_32hfw45B?|Vj(P-4'
            b'lRbJ3wjGzxvObCq#$#9}K>C}9E=KF$?E2ly7T7Y-88i;)L1kM5XyD<XcVZzgHNnpKIs!b6#|8@mXQL!UE&zcZ1K1141>s#2+72O*'
            b'SW5W`3<XkjDwi}tp}z>{tP9X50-X44NhH8}wbVjkyamJ&8wYt8c}8GU5JpRSGn~Ln=p4{Sfc8rwFPYxL+B!H(9ep=|{R;|4-Xeqr'
            b'Nl+T!FdjbD5H%zYr>viGSEOVebkBO?59HY1$9LQMFbZcN2k%+1pbqrI3{}H6)?j#0g{fw6rqeaJlB9V-2f_@z^AitY8jmHqGa7X~'
            b'UX4J2!zNh7!o*TwYq94=Mh-614kzelz`(aYf#X#{oKOpSvYINz*Q&%ajHmrt7v(oEfE-?6(G(m&0lYp1ow3+TD}LcmB3_1+3`ozL'
            b'5acCmPVg|mv`Tz0RaaZmAIc$(z%4{Ce2VmkvQLuj!~^3CJg~FV^U*OV?jg)ilw~^MrUlz<?DG_|d5hBnlf!dVcNqu#ol;I-H1o)u'
            b'HCBQE+yXWg{)7i6Aqj6nAfN;MD2wTnd@bo3@y>@dA-ui;2}Sm47fE1&79K8-g%H{@vG!2V>qP(oM+hctY90Os6BR^_uZ6Yi%_H;|'
            b'aWX#u0-AWJv&r8EtpnoF!|89m#R@jzpn-~Gh$;?^A<!hl-XDjk^Nufm>Ggk=a)-p9iVOe|xhUmH0B!IuRw#j6pM%vAdoByuEjRYN'
            b'=YIhIc?L2YB{o==fHSbqQM}i~719Hm1Hg>nPzMRC80l=oQqfOvaB-ly5X~S*5}&mq6Nl?rpmyfKGbs4HdlF8dkUaRxrU8L5CwCVK'
            b'V}gdWyLEuTWh6%mYqIYxG9(0<A|z6_h&#j@hZZnS6~LZ=6#=d>NfLtCfQxlL!x30$Z>>NfYY!gG?*OUJ0b-acb63C4qis;~u`B^V'
            b'B8+lKiSNXgimadwHJ5=E80I37{5BZ&ua1XT{hpXcA(<=^Ow_y1VjKIhG%_q5oUt9+!P%OlE<AF~!WO*<uxX|q`To^+7)_z&QCx*A'
            b's?evnh?Nbp+&hak8=GmYh9tk_1g3E8fV8dD22BXO=z2gSL7iL`KbWBuX9VG3VCE_!aoGz3X$^81osNaSV$X*dm3#-%8<rHfq4sh_'
            b'&dZ6V6o0K=_7fsNU&h>!L#YQE|0q$;dxHVMeHCEzYo4kWJ!BhEG#-P>z_7}A404=5z>}7Kgr7i)t-=ie5uz`?w}+ns73A?_1PlC&'
            b'TlQ^W3lkLiNXi>$aTlAhJy77?!H_Iako{yJ01@g?9dj?o@}$a2DM{(+n(++X#b92um=jso0q!DX6c}kdegV3F5A;Wb!O{3@l%V>t'
            b'&~U!!qhkY8qho{I&u&DtW-J;+YeZndM%V~OXiUqsN+O{o{Y`02gyoK6Z%$&{9j(xlkWvv;FX)^~r6NO2qtR@QKH6v`{N)OaZ$bSC'
            b'A+cr2XTf^8A{q%+^1TLZ>Ik+DJ-+FP#NtR|1j&S7vng70cKXL&e<UU9phnNW^e|pDLcH`w7w0{G`T;}Bml#?8EJr@Wfw_@BesGrl'
            b'V#^O$a>430stawz2<*RE!VjcjGeVF@5q1akSXH?MF!Dy|OzZ-DG`Q+t_D=`BQFn0611g_*^tty-@3UOY`PJFz_;VN3m3(k<axxfw'
            b'f)yHHKVN*|Z<n9Cuy9Z^a?zLli}$16rF^x2b)FgfYV`i}Tz>wD0l5M4<I~sh0osmr6YSuCt>esd9L^ezz%~Rj=!k22vtk4UBuEv8'
            b'qcZq(dU@G9!9D$O`cXf<1pV3le30)9_*0v&o@Hj&4<A}*=m~Tm{FpXdHOwAbqZw`dC<fy%HlHiXV}%SYH<tB@hzl0->dk@m<@99u'
            b'2|gb-TbA_$D7d<goS+3P{Z$=8u`z~Qf8rm@;#m3U;=_kQZ#Za+WW8E=*ZjAS<_7$AQxlCw;|61$cD-2>`(Q}gV0_h^_@Svj)cI2t'
            b'nbUS3YCzvr)9;rd^wbqoOUOlAVvsl&7zXE_S~nOWWR|8BT`b1Dv!O=oTcJh@My4UY)?qgID9qqvcd@W2sOLyC9DS*9J!dO_JkAE-'
            b'q<W-~KAH*WMS;OFoB31U3#MC(RMV2IIy#9^BALR2DCk+vee7RcT@o9?7QDe{sFdcDW}E#g=~g%2?$<$=tjIj^QQa=|u>xZU^af%='
            b'd)xBBW9g&D)Og0RNUPF_bO0iQ=x9iOI2{g8&p!(Lu-RxGiuY)Lx^;A^oH>T};H~|G#w&~y)WJ~)I;x`AfyW563j{H~BaRLlt$oln'
            b'M~98quN$<|90$U3k8ru;@jkhBXrtu9GZ>_NCqi_ND7J;}>kDDuh8x%fI%demcdSBu2ZoHItr@5x%LDAZDr)r@Y0@Fz1}vIGhO;PH'
            b'kU^gbDjgoKDbN8>>fn^rkk+$;RA8z)6f*!A4=5tmWdDtWbe0Ozp_TG;N3>L<oUG1@SV!Jyk%q8IV^r+U*!|Yig-j}JP?AbAX0?DA'
            b'dvB1i3eKQ~1|$Wxw`onF?^7DyurlEdj!PI(4HeU%cm^ouIt3<N6@LJ^G$Y%CA`zA{?wvF`kcm#9>Hd2EM+TXXl{{zp6WK6)$Si5u'
            b'P#bN5baf3~Yr>GN;p9)i?Kdb{|I8X8oXsk260T@wRi&Qz3QU8K75+eiIm?Obh*?|AuDSU}b)GKbk$S;U>|oaIcpYA-+!)QkKZQXj'
            b'V3w#sEp=3tvn9+Kbf3&jioSME$M2adDV;dGtmlDY8!OJg`63Hs^~q3D$oX4vmI#ffgIe-3K}0h@0o1=Dt9>C6T*q@)_J2YUe=~Uo'
            b'Y6eY7f&eW7H(|6ML=K&}=*Txl<owu->9jf6rQRaWAPFXx>llx?fREkv)XO+&M({(59M_wPytQP5sQEe1KsRm1K1y6Y;7=;*4N5y)'
            b'q=^tJG-nze+F)7-i*R~}Mh6av9=r@~p(Pr`(7h0pUTY^&%14Sw|6!Pk34*{Q3qF+L0Z_Wh#`8$##A)1I26ZvqE1_hmy};NLd4((+'
            b'X)q~Iz+9k%UToQf%E3a}kH;d)`^K~n44d&Okvuzd^)Uhs$gvHyWH=H`p<5aEQ(;9AcnyVY`uFMnlevwIU-}604!6_J?sAoTAPGo|'
            b'@L&WM$qKw!&z@D=nnS=PT1P>KJ@SSFv49$d+2hJ!jx{ky$$X0W{|@tgL%5`qv+#K8N}nJI9$TrLOhWp1nCu&v$j_`0P`=(<G^sn!'
            b'1>swm&MlB8c_KZaln5hd@hvD@JPw^}HW@1OQEMj@PevbkIRNIH5e9(EHV$6Yoy<2;LFtAaFRFDIuEvX<1E!!nm1rLs_A!Db!O&Q*'
            b'FkGB!)j#V`B$Tp<LH(P~R?DHSTFbthK~2$MG6%|Z&5u*WNX8B*w+9TQ>}A3v=AYuo***p&e`Q_&4Ad3GY>Ca7qfZTKQAX#HQV^<7'
            b'z)5ks(twrur!dV<1QAqOX<J-ysg<VIoPBx6(29A#z1nh%FpGlXg0s^+WwiyN`@OIgP3>KDyJO<Vp)H8r;`X%U|9LDGMfT3Y6)a$6'
            b'#!2h&=)1^r*SZU{6uZEb&mqxgfcY5h?job_0#k_^L)=RnPh0TWxMryhpLK$F81$PmpG8~tA!fStUx9%g+?i+DU_HAHWv}YfPHA>#'
            b'<N4K0pKpoG)Z1Gto$)=J@}`LI@AmuN6oz^sf>mSb2mW%sw6#tZKmROV)x_)S<J5;M)KvV`MHU(4$t!$<Dg6&vp_WX~+svW>{EqD{'
            b'wBes;#wET%7lkRRg(v@5w369NXl)*S7hVtG|K`y@9FGTC!~GGv{V_~7kDh_bR!uYypO4G`DWM+xJ&CpS>Gz<2=Rk#Rc|y;!?JD0g'
            b'-nJq-teq45-(sw13+Qrsc&^y~|4a-yT<oR<!#n@qC595KOPMWTcyY9~a0YGf$VsT~3ZPPgn9f|>b<C{7mpa8w#!tU+6gZAOW&3r0'
            b'3TLyQ#$!&w3km^Gfu<eh@tLV<ylNwijCaSpuWLWzBc1Yq^83?a{li7S4*wddt15abL)9`bjjj=JWEb$2IGllK#6sD^47`nTC5{s`'
            b'C2W`#98(9dM=|SqZlvfNX+3z37g5?SWxMmaH7sD^Y}3H1j;@SGKpDwsl(H4x?MSIWUZ+9ZSwGuLKN^^C<JlA#y>zOS?LirEBet#E'
            b'VuI1~j`-j#Oh;gE?{4En^H`oCpaGAEQfZ)v!neMMb~V}aHurC=!j)Z7jg^|PD>9>b^d}_4O08N=871Q<z(@dcW}Hkr3|0vMhKdRn'
            b'<DVZ?+8rL@-8&1)F61O>7=}_2@yZ1sPM;13GPg>eY~x8moF>MqA>ELU<ncg1!w2Ya0BOlPKXOt(PLrWHiOt~?2ug!ahr}5Dbn2}V'
            b'O(tqs75m4$0)!JUr~$XkfQA4Ul*~MoRLp}p3up@Xa$&tLmg|JisB=dODvNBs2q$<Cm9HT<5$>8ipy2T~;d<es+jPfH6f=gTVs_Eg'
            b'c_}d7)Qe*y@tbqg+(t@bGRj{(%F$g6r>x9f)cyP!ZG<AHc1Acx4j!ccg2{=!C<-IH0-Oztp!-ywJbo}mN5v$(I7W<h#5LbxL(!Dd'
            b'G|o#i`_i|)F+q|qe*>CXWgSpTND99AvwZchm1=fClMUp&3LQ_LCL=;|Dr3;!ML1=;Me7wPhiXHM8<HzE4QZ#ywY4mi9cIOv78K^_'
            b'cpc4<57$QWn~isy;Vr`IHGTo@PaEJ+ey9fFCOcea@)33JWFgA-4{}6R<HzzT-E8e>6klih-sDGONjU;lg{vtY9#lO&r!&COWIeru'
            b'&IX5_v9{aCV>+3vOeyTlX_OYCG$QoaZ>xj|x`6h+_Yvd7jww$vdEuC8sWXmLels1t;%7MtUal4nUTdTyJjWP}=%+ppLs#(4cuts~'
            b'IX(&^HK0sM$E0HvOIZ+7_D71sgf7^3%Eq4wKo{*7D5ES@2@iCR*oR(70W;z@lXh}D!HB~>022yRjbqy`ZV_g91$JhfDp0SRY*$_f'
            b'UWAE{4pR2RnN^loO{F>m&H~UKnA>d2^;ENbXaaQIuIV9e7&@B7ri@OT$J{Gqjhbo<uw>|NB?2;8Bu^CsaqOi+z!;Q8Kn8xJRk4)u'
            b'C}T6k{%@Y*A^a{m$b|6Pc=x6_k!f|nn`m~}`R7IW+Gggg_zM{!c>I>ZhACev@DhlnT`*S_s<Dyj4~MjhM*I)rBEVd@6*L1&fWYKj'
            b'U~%00{8=YXXcM?bm(q-z=nMbO%WxU<_fN56)R-k6r)O&><2xpPf_4U$exQ?mk`;c#Yvd#V&TiaVQdv(<T=(N0x2n((F@`Y)8~{dK'
            b't|X8bZ<q^*7iO`=U^U9A>ZS|hCSypt!w1V{fvA!{>0*d{1u53U>juHp;|z*-S-j%y9Am^FRC1EjEfO`jK=zfGrE47^z9<`kjT^A3'
            b'!YN%<@D__WgGNf6#|%fj5yH6@5#2fg)5^tzT1m46xW0bnP_`H6P&A6vuIU_v!EC))*hT44R8d-R6huW{ei`&?@jA*ggLM@ayWbQq'
            b'MZcwQHSfydGkg&qBfVW4lT-yO0i3`#0*mf@<ZQ`1;Gll`ux50)Nre}9J16Q^Ik|SEYB;yiS0y3{`^wvN3ki>nRBrak6UN&gwoI9<'
            b'rJ0hL4@g77knydvLU-}{006sT!Zl`11#z6aTW@CdQdt_E9LzVU169m|uRPO{ZJI1V7eDzzJy0s%r}HQY$EOruFtK5cTS-*wAkFcF'
            b'-89mH^>~UpW4=30m%pjbg>lVZ8tQI1#qtfoykX*Ae||vI8_+9$6qIAG8@8(9<;z@)%1JfVFRI#X7I~HFqk;1E5w)8x^*e?`-I3Mc'
            b'@ghvX*Hnq;83mpjFsn!dM8t!5A>4=TbHoD{FLJk;=Q1e_G!VS`b4Tn~#qa6!KQXrW>(*%6LKEloX9GlOYA|5{uRB5KngB^JuK?6V'
            b'6C-1M2jE58>47@&W9*X|RwMQraf0TE1o`(OEfaCAC(nkl<QD5!VM@w*wR59NtvX#t=mHy~HNg9V9kTU?)2)Dfc1V#`nWO(1bK?Ce'
            b'(D7~PrgBKfJIQIc^8{&%B4fzx<B5aD3BF)HPlFrjvy80bjBa*AZ)F<4y2Z5hfaJ)6;Jou6Y$LB6jNSC+KM10R$$S)|WcSmyHCuHI'
            b'Ag}qNRr>~MC>3apc)hf-j#FpfV5ML?GTBLgZa_<+x7>nsXU(^3W^htN`9d^ng=jGn9^M8~JD#QO9cpUF1{*~i<Q_kLYaiRa6RPMj'
            b'H42$LC#~L4Gs8@k4XHxgGSMC{3a7!uT*V9CYrZwAhcmoBHtR&LoW{E}Q=dmCjiIY+q_s2Dh$@z@uRL(&gt~&qxYnA>W>2eyWP&4w'
            b'G8@NeZm|WYY*rxA`Ym2vQ5W11HY^{dgRUf=sRWkd_-wrZ(*E~!uJ31JNG7*XHfoS$6oq+dO%2{49e>@t`G%fA?GRzotet&v|0bs!'
            b')^VPrSo&o;;i%ec<Y5iqaw8hcxu%zH#9umU9#9rJ`P<=<(<QiypgU?9Qz8_EZwb>Y=2j!s%8ms|sbn+$Ms!^%7nU~fXJlDbU#7`D'
            b'R@bS@g>ZUBu$>F=M$rMl0n}myT{%p8a@y}vNdU8af_X7ZWjpg={C;{{8wX<#feWiR+CWD5(;L?TfQr#zX_T75SRi2!_lc?pn91Vg'
            b'Bdgdm);UQlqUA!r#&GhcE8Q=n`|OmI%^=b>@`*>w1HXxaIea}Bflmb2U5P_uxcJUrtw?bt?21JmUZQ356CD%2)J~V)&<8RMN%VMW'
            b'?+y2c>fCfnuVFS!Iw_EcK@tWT7|e+#CvmtF6IIx{Q^AOAU<DK4Slcu$Qr(#wCY|=NVe&>rEn}f_458=`A=F09LfR?%g`DA19NM%?'
            b'(s4)~>OpoYN7K_u;a@pZOor<4UN!Qu9IYris^^dS(^*4m#7Wu@{DGYv*8%jS?mrrLfU^U=eAy{$<Kx=&^;Yq%R<?8ViJkFeHSH?z'
            b'oTb|wt2S7eY)q+9UPje-v&`f8N*mwK9n-J*`Van-Ymq0{Z3%o?YfH7Za;<<^XoA0U&yA$o^2AqlP7NJ0q$nu}={7@qJJKcb_0m*R'
            b'nGw-uTFtioAoLAY4iTeVk#>o^kxDmJx&BS!+<5_(LJ0!X0N{b<W~s?#h}TAqVw38DpRJt49x<M1<VLfntEl<X4{-HfW3_F+dR2dQ'
            b'P;0$Dthe60uD&xMjAXtckE}#dZoCOs<nfh#DZ$_xQi&4Ast|@UFelsGFlH$Q66ahmCSM(u0{C%bNA7uP7Km-zrjK8R4+aA5jGHeT'
            b'iy3!W_Y%4{-|pwr5_C*!wDJO%J)0ck;A8+KQC_Y^G8?wakGUQ|LjIIGT=S`3168Z7dh-F2I`7M$A>_uQm|LZ7s)+o_P&)9VI-?8;'
            b'DjybQZ!R*IY{EK6G<99IXWpRIcRE5u@d#`Hsb^`rwqZU!<`kXz59YnN8e0{*BvR(&$+V-=WDnCgH_9_fg@u)*M(JalfJ=!eC6Elk'
            b'q#~a3C3H8Rt7KA)J|#7*kzpt~s+9$Vo6f4cQZ*|h!lGV%DfGt|auk>XItOdqc3swj3Ov>I_iD~ARd<%tGqM<bCYQW3bkZ)qoXfk+'
            b'nPN2#Rjs3!b2$R}=(j9s2bL-&-WYZ0WZ8hgJUndV(R^L`TERdLp&@(RBqP-(hO^E}acf6-z_7)$6yDXE(Wl09WTnM3HtHX;St(zF'
            b'?XG};&qoJf7_GKOrU7gX;<aP0`ga{1Fq4e1;+P6`&YuVZ3sGQ@fuOZqbI%*B%9ev$HD!<Wumo)f|543&u#5xa86$2{u|LM#9d<nN'
            b'YwFBOxd4rGHJT=cU9;<H35Yl^*Wj7qWX=|$le}sfuA)I8IZv@5qD<P!I*A0}v!Zm56>yYcaw*YvmKQLBJ*U~4k!3{`Wt9P{NMKY<'
            b'L{YD}+zf%Jdlp7{Fr$)zE^{f1jHV^Vj4A|X!VZ@Mb2(2Bqyv^kR7bwjvn`e~RMwHYsGM#sU09YExYa^r{@aeyg|c>Rwv1kmR?>oZ'
            b'R3eWO3aIQKTID5O%40uL8<YvFzpF;kY&rVM@(1J;#Yqfiu9av3P8*~*Vo(9o+c?B({dTtak~uu!4t)Pk1&huY=kZ_JSy7TtS9{<?'
            b'fI3e^asNz9G2~O`!_dVStr4`nbyF>LR;2?_PMCYjMtrAeHK}v;C)!L#2AC$$j^5seVK&f=)l!v}O<=ULHzs{IJ$p4OYT3Dy*=vms'
            b'HVALjnFgKW^E=gC$^{kRirZr0$BA*$iO9OnOVTi*{35l;LdbBGh6`z0sslE{Dk3EzLpdB0+1Dts@oO>JSi#qzGe@_imtHdEKX7p|'
            b'+Ul}GW40Ray53Z{F@Ve-8=0GF7B;ifmYN2zX6{nS$W|%5L=P1{&w@{e+4RBqNe{LlRB{FtU#eq&9125yTA*Py9MBNcTbVa$hRmd-'
            b'Bx42SLB>B*63fQ;S+R&7md>j-%2&Vw)N@B{lEH3bZ<|5hGb*b+#?`7ebH-5FE2A_eO5_{0w2+%o;SDumNA@U$i{<lFy~}E|_D+>9'
            b'fBB<uK9e;rza_AN)eT`?G)y`9sn19BGWAn|R#kD~REL^z$*Li8)!9f^s;S>Dd5RFjqOoA`>47?_#r4=8?y$Z|NgA|{5<=o3yZ&9G'
            b'kbkoQTj*&zp5pbLBc}d4-+&yYg3`Xl$hGNC7~iDqtnCmj)nd{MN_n+cW&B+e+4=H}Q<B80sWL-@rd3*8)Q<Fem^)*Ors}!;F|EJ6'
            b'Ng|=xSi%rs@eK0`VVG;_qsXFInOVqY?olCN?I@}gwTxf>+%lKyPp{uM%R8z+gJHinJpP2oT4$H#O(k#1G^VtcWUfJ1l{AisWiZ!w'
            b'*OWu5q$HAI-u1v(7T9JOveIBbatQWy?SRCH6~~is%%57*TypF1T9COm%w@>pA&ZuDCA(peKS)3TfPF<@ZQ2;juMW5Za;G#s?2%D5'
            b'tr$n7J5tPc<>R0TblrxN*lIAD@LfxHV=qNvwIIy6JiGs!mp8~v&^YBlBH?GWW_A0^m!{dxU1~DNz+fuig{)ndg3FRG%G++GDR~@T'
            b'$hEwn@oOe^nj~zd+Of=0DQorb`jZMMnNk^f$dCk9?#4V>oJ8|CH)gFBV&IOJgxPVbPf|EfFsfB#g8(SzCT4Xr%}sKftQ>~gzNZKX'
            b'jY9|KI!v3QzfdHHhsu;9l)kvK(5QIdd>yHuYIt%f!b}BNuFsd2;nSCw%#b=CYJfcb3A|Q3L;-_VR{}G9*W9&<v_Vy-P$w~Fo38di'
            b'MQqFld1|9eca*g0vmm1_S_h@?6l;N$LRdDNT$idxv7_l=|Lsxf%8KuC02ND^l~riVUpdr;+8(K2W6YZ?^inJA<6&Fg)O+B(UOa{p'
            b'C%C)y*UkOLn*;G4Gd6qt<n8OmK})=E2>Es^lh&<Er|Ps8b*mALP%3w-ZvS2OFaD)>JnWr_)?1-c$;`pPbo7IFG)**ZiL+jJaMg$I'
            b'uUfAfhk_33=?3BOK*S4YdZ#jKa1%`I?5h1zvmPnYQ$(OT2P}I{*J;^__fl%vS&7ByR8mqYTX!*gF|hs3JM*esqj^0CUX!P>731at'
            b'4whb-tmGvlG#fW1D_ZNp^I^1BkLeBlh{`VxRk2YmuZUipXeo~xF;`}9?{EL{pNjtW_LP-Jc^dRnll<$EmetkglnGsB4Z`%gqs&X*'
            b'a>AmXldiR58Am#SrR-+>nG!lgNMvbtek_Z_##@kYQ5dLSs@XU`-!cqkUNfIn@PLs_V&<x-V8~ihR-<gC-~0TPf9|t?@rBZ;F(n#*'
            b'0qJwM|FJh13eY5UnV|W$F4Jc?AxE<pVn*^|O}u$6-dhtdNxZ0L6fVR;<-An6P-HF$XNny2uhThE?A0g0M~H)F?Qp-<AjZU_aT>Q-'
            b'J9_oD@%oL3Gru5^{*Z;r?{T2(8`z<kBe8K}P1Y{{0L|b2{l7%>Xur{7?qv?=U%kK`!rx9&m4MaHG+^0AF2p(TBD_ll+d{v#Ig|Ax'
            b'2{l-!G5nJ}q^3*FI5O_>-X8wQW-Nbc0#-q5!h0ldm9X90Ygoe#kju6nV39~J02(Xvz|nAKu2dk}%{!kQY?e|?{RjlzHd*=8J6R)C'
            b'p(>MlxTUNnCr&pdrln#mg|AstB9|Y`yeLvX$e@4Mlw&PeR}%>~reE(gco=^IBEjFRQWZ(n4@?z)i8h`7y!Wv?JpH9dmWls>xQ;oL'
            b'q*5DWVZV6Wc-v&n@S+KHad;%&SKmqEyTiM^GW%ZsI;)!3vGhEFH|LAHGJ3xWqdSUpV6|-IU;S$i{9|~_kj`Q8x;b7|_xO&nst&F$'
            b'{+o$dcA*SFdfopd3RiWXue|Cbh+GN&(S56t84z#(NUmaO(kQ>PE|b3W74;HNch|+!y}hhWR<-^cr!I)N'
        ),
    ),
    'fx2_lb1.py': (
        '460490e427e54d89f0a074d785cb8bd7678df509215be1d2a37a6f5a6f617a75',
        (
            b'c-q}PYjfK=cHj9cu)34Ilp`sYVmpab-Kk?MiM#Qml6<pwJg!WMl+3k6DkNoFyWN?7?B`B@VSh=_0{{V%a_n@s(`n}3L?S@o;NZM*'
            b'a8#?+e19>TK6FG=WH*6WCh^!Chx0I7H&4!vdS8V15T^Sg2`4u}A|~D{_2yz0#ET$H*5W0-G#K<=9h{$tH1lRb%W=MbJ{XGO=i`Am'
            b'I64}L;hDI;=4sm3*9~!SdL+Jn{!R1_4nJd$<Kgkyses-e`@J5#ZgfU??gIZ2&;<N95AT9~;fZPZ5cthy97dVA55n0^7WmGm-q}fS'
            b'*grlLi@=LSJQcC;r(&_1XW?=l21zQiIKCCl-M#JB_8aj5x=d-pXm5KP+HQjgMh@n_@G{{!<1lN6eh_8h1Tf(Tlh{W9^KLC6j<Zz6'
            b'_Yo}nwD-s1==kHP-y2+<^p0Rfk+;D9_p!5z0AUHhD8ejQqz&OkxKdcqBKBAF0GBq7vm1mN;8}>}e3cSplQ_zPhpg#`=>+ft8-b{B'
            b'uqj>!iI>HR@X}itc^W4GPmlU6g2{~+h3P`TR?_IRAPQh-MG{N`z#;4}EC#-Lei+T1IGV4;!IysT;OMvJQST6Nek9WMVjRx_*fbX2'
            b'e2%b%8IF|z#Lag5t(YbO?8cviBo<lXMd>2Ua399&EP&UdSj+=2i4X_g7#5k{gv({%Q_CG7Hg5U=6K*=J5)c$+Koad1&{Izw9$X9#'
            b'zKCJ(<owIQuqVDA48+ChQLo>;7yxW8>~OkDcre(jG+hC%>n2IHM(NVaLT^5r2Y107_5)`DT42O}>3tm1He8&GB~a5G_7SiEC>-9v'
            b'?%#VUPyoIWBzG{(9c;x-98LnU@Bnf5VRjR*GQnT~3TaFx5fLY_E*eO$*}3=bf@uPqi3n`J+t&P*#H(d0e8fW}#%mxXdNc|nKX_=Z'
            b'-Xa?zn4a)M*u0FGBXEdN0M2O$h;`MXJFv=ryN<hz#KJ4Z?GnL*H_7Jfea8{-ho*2J8e&}+UEw~6-;2H1b>Q)i*b?rVzd8;8yb$*{'
            b'AP|@q31nbOBn2<Rf2L^wFvRnDe4irpUIK^#fq|6fWhh4gN?ez71LO)gIS>)D>o{H|hzW0wB)HbhCBcMA0UIG~6+tQp8Q4F~fQX@k'
            b'lP-WH3CkbjgrK`PZ}yKreI6PUTwl8nFWa!nX1jiUE!WPA6~F7^-}`_I!ocr&{cXg-mb<2}O+W`UikZ8_FKu}FALxrYPS+eq{@E^#'
            b'Vve)7tH*JGwV0<QIpDejWY6pCwT5#WcD31*8v%WLe>^xGe)&zw2S3chI70q+7!NrHm~<4<KvH_QYD-El6Gs3(Pb6mnJJ5KrL35Bt'
            b'UIeTTTk-B~Ywz98?u!O|X(lLnjYsof%3oh^w_lLJh?)RL<fqOuPHCzXrr6p-*3j?g8%A6=zxUSg5BzuMg>#Y|@MX1ZZEZ16-QUFX'
            b'z*zxbC31vsE6)c3l#?O?o`U#=5!b?nnHaJ+t=BsMru(wfXm;xK{p|~s8iSZ#0pScQ&5*xA?xVVZ5dkg@Awqu`C&-9VkOJSvK}r%R'
            b'3DOKOMoj7CT(B<!639s9Our#W(wT><0N91u6BG<UdVQS+^J(ig0(Al#z5uD-Q2gKk0!&@W10a8Zc3}Z+V37m^Om=!U6l>s5#ayUM'
            b'l5o5N>FfZ^Hz>ADF4gVWF!Pcz>Uc)E!Pmh9peO|7+~4p55LZBY3Ux5?eVl+UL+~bXn!Tr9xE^Xl;1)oKDN=swP0@_#w8WRQLy-5M'
            b'&rZ%h^-g=o!{5+MI1nGsK<Xd#e-lT&VefEw)+g=2M#0hX5v}p_!LMjcu<8pyJ<<ZC^cT~>%YdFyOQxdc$6~QI_EUH>P+}y{3C#0;'
            b'@96jt&6HZpIgO*Iyhql}8Nk)A5xU|O5eDiYoK8XgMw#<%1>}LmuQWeUXMo%%aE6fSRhn9W;uv`?Bec&J-h;GBK<zI>P~Z1q8fc9v'
            b'H(zlV@<tly9XH_xfL0M01&bI=ixf8}j2ur)0w72*qd*g*DM6a%`=fz5!U&*&peO1|KU_51JRmW>muPiL8;_<4kwTcJ0n$b^0ptJ$'
            b'B9&cVkM#H>&=!jrO%_l#q_}}c7)eLRgW>V%;ZRKG9*yEHVyS&|aUq3i4zGjjLMa<BmMdf(H0qK7L?Wp;wLriq5HN^Zq~CZJM=(eh'
            b'OOrJ4L5KoUXbT-QeSqd(v$X<dv3MoWFdl&M1V-;*c<`wwToSSfC~PX3%rqA8uo$mG*it|ANqXFXb%0{%Bz!>91u(}U$QLj#7UO_~'
            b'&T$5-im)%3zOY5(&`X)oL0I2`@sLW?KJ1H-j7SGC5s^*>(p%udQk1LPt#@x=zkzHPDBEW!dK67&$toia1D$s9NfMxa<s<>y*a#b2'
            b's0uO|Vd;el@^V9r14MivMIQGsm3#)2f!3xL&M8>0YbjyQ5B?e8Nblt<ia!`@r2lh~39Af}pHp%6_0&1L7@l7YanlC$1`wSXkmV|R'
            b'yl^CBXa)i2`QDODxrMjnov)ENqA`I2=8oLvz^~!Pb1O*T6&M?9*d<UoV9tVMaE4Md0w2%>MBGWRO65jKyANiXfSg5qNL3#}Phvj!'
            b'pQJnL5im`*=r!1OTt3VCB-y#XhGhbze-6;a==_V@ynWRHTLwCV#sNL3Y-<1wJRI~+9OR`o*csnOfT!zgg9U-JQ4%5-fIv?H?78QI'
            b'@NNighY(0CrNRt`0x3F?OB$olpT{%S1?Uq2PC~XMGGM(%ZlN&VBI1a94|x}PMqpDECkuKr9>YuM9MDF9_REqWo7}+Kx;RS{eK&yp'
            b'D+)&5B7_A=P#WJb9zN9&wImKFte<gLq+}f&ob<#W$g#akZ`aLX5>G)6-mzdo9T>(IRl_zmV0ci4xn}StlNGp<q<KLH!VJ9g;{aj0'
            b'zLw~=H0t&Bas&b#Ho+klCYAzQi#;zga&Vb$JVrMI2EGXioS+Wkgj&dx)l@0I)g_i;Jnh%|D8G3D<nRKE=HLJd;PnaUjQLtx@$+z;'
            b'@G_)iKzhE9L0+=v1P=pDt1Jw1b+sn_p&Zf#+(PuiCrE!N`y|<p12De813NiB9UX$=9>V-YS*8<iTCnZbHcug&*El^eIXp*oXF1^S'
            b'm2wJ_X+Y+zu@VH}2C%8{Cju}DNq7?i0Uh8+IZU7AYf0CLcRr>G;q`lvP-LGTAPLOT!o%gU5JFof)gB6Zy$B%S2*HF+ZNk4`qJpRi'
            b'wXpVsS%Ur|P8LQ$K;r;)Hu>A2bwC^jIQ>m9U&1Erv`}#jQN^J#1e#>n`{NLG-r?D=z5Z`f?vVIXkpUngAEi7ApnLp_6-wkcXJB=t'
            b'fzJYV&5iwo(?5X!JO!DJ5*sW_z!})*DBhd#66pcW0boXOsH2QkjC8hPspuzoxH!;Uh-Q!@S;$(EiNg;ZP&+f=85DfM9SJ8;NCA9h'
            b'(}2L3k-Lk8F-F7LUwc5{GLj>OHQDzL84`j_5fUkD#2sRdLkk$EieOK`iU8M`Bnd%mz{NVB;fNfxx0axgwFeL8cZ5{u0WnOJxvO92'
            b'(e6?5u`B^V5{z<4iSMP3imadwHIsoA80IXJ{5BZ&FAj$n{hpX4F_|n9Ow_yHe4U1=G%_3=oN+zc!Rd;lE<AF|!WO*<uxTa%`Tq6y'
            b'7)_z&QCx*AsyL*$h+_v??!5Vmjm<n(Ly})|0#i74K-yMngC+!C^aG%gs7bC$7)?=%GlFn1Fmn}=`0NFNv<5khPRBf4vgbpLO1=Z>'
            b'4Mz&xSbI65;N_%Jioa1WhZzxIC}VEOq0|G7f0QVvy}<zBzKAgTHOp0t9&#U1bbSpf1H&rU*C5CF13c;INB9Y(*fPEcAY$~zx3=(8'
            b'q=GzQieQ0%al^h1Y+;5XA4&P%o8P8pY!4Lp!C*)hD9C;?5P%4EsE)apV|h|#<&>oKbi;Ut?qW1+ILwKx>i~BNG75}zef<)2{}$+v'
            b'1cRgLw<JUL<DlVu)<?$%rbfpGxu4yLWW`uCh}MX}f{k!57@;XG*QtnvlJqyFH4&CSN`o1RZGW^xPeMvXP`#jYYPFgSF^xvk75ZqS'
            b'kq8${Fun!#BZQ=mBcDa9#gb?wTFUoYu&E>1I`sG^BNB@vi4i0de$Az5&B^f}d;O7=sDlPQ``W{J(FpO<8=al@_~}OsEuUj#`HLL+'
            b'1PA6u`uNdXg!45&V95omyH{OkBSv8VO*4KV4Vw{yJW8-TpvS4pC4iAPLT6$V;G@As|Ga-Z=#35rhdiM2iAP_0zxKY!#hhN8j1IpX'
            b'fVz?o&W?@-qtCEH<Lj5Rul()&^8qXzl#E>TdH?LgsCO=3?O&W)V_%Fu9G}Y1pD-XdKz@As8a_hXp>Bd59I$n4J;&jk(FklqAcL;B'
            b'q&I6uKtO_2V>oJq&&TKIy(8SykH??%({s?D2VZuIodJJp^VKtJcKz_7bAq10!9$qRW~+wjLuWLljUT08{H5k|MR}}{q2<bPJ`-`l'
            b'LSDb!alRfO4L`%@-FC-uegXy8)R7akfTh2xLntoBaGOv3<2W2EADw;tIOq)r`yyK{=fNfa9iq7be_b_1tJS)~Sf|@;H^ep=k}eou'
            b'%{G2$s}D{7R7d7?{f7q7cir^+We7cY#nciC(Uue>&KZWmxu@0*MhKauDMc5{F>gE6=zK5KNWsWF#MjxigO9=tJ|4{H4h8i*X@;XO'
            b'6|ZJ&<zHXh0XV51DWs2P0(wzka7?G+Bn+a-+9B1nAghi}B9us`@E{6$mUEx_XBX$hMz95M@fj+m*|@#WewB2q+wZoUpi7oyo`k4w'
            b'=lWQIu>*PoF`>O}dEl`KQDbU6<5;9sX+$~zkwJ7cBtISxhsUR%guB~rb#}!Ev_JhOx>Vi_LwoSn_D<_H#tE9>r~@6<(d)ot1lk3H'
            b'7~c_lJFU((=$gIV)|)piT4{j;VYz#_-0SOYa_!JYDTHS*NcmpG=p0dO3*Fb3!o7*_VH4<>!H(}ZrT7jEnIvm7P(zjn*m-r->M7Es'
            b'N4^bMG`kFES+XF5J`+?rJX}$r1EAEwDXAfyX9cOiRP`uk05I-QM6AvJ8wcqe6{JHe<(IDLs75(ilNGU!ywM^JVUx$G*qw3vou><#'
            b'T-cx_m1N9n0WtQ$AYm1pK?@B?3T$uFnn2&DGQ8nf;SG*U7*P!s)1Y_;DCQ;wCVUlt0J$_J+k+wzjxz4OJUU=SC(v|%v;BfW=3}M6'
            b'Tl`Em3?DK}S~k?izCgOVgsu%?$kurLC*bxgl&pVYjSx?#wS5w<XlB)=p7;h#gO3&dK!SOTvG0lLzL;Ke^Ofp6nWrQ5f}z;KtlRK9'
            b'yimI`nt^{xgO0&0QG+__sH$K~m^0`;)=Y}NHcrRynJcNB*k0E2z;KNf7vFqo16hAEloWFQ9-I|I<LRJQyi5?$%ufLIugGd&Nd!0X'
            b'+?D;G7{uRHo`IS{Q<5M+i@;47tp|}qCoVejjS;ypHDfya9PCnW5oeGD6N^=fM_j<i{%R6fPMQ(?kRrzqrlM#qxgctODKgO2K4Tvx'
            b't{(7b74-(C9nbSb2o;($jSg)vt)h86xkaM`2Sg8EhPKcWjZ)}d3QDiElPKjQMWp{Qv|@rF@W_IXWq1ITZgw98By-X{ZZ3nm819u&'
            b'veceq?1{WWmW?!+lqX;=&_OS@+{ennLfMbUBFg*5v=0oM@hOo!JM;B10u9Kq4YXu95=^068TV6RMG$x`g>3rw>Hd?sjf`LV2=WfM'
            b')3tZG&OML>q(!(h0*hn`UTok_>-(BRz$IEGk;NWG!+}^pjl%44VKB#vn4@Ao#r%JR`Mx7u^2u3vJawf{5Co5{R8A%#{Tod79ZVEw'
            b'RtTtG?+u#N-RFYvJxmuCNRvE~9#Bfei8ucqlpP+2&NbT%mHDW%5sD|H4<1c!+V46ejQ>{c8@#9bsP~AmTB{|7Tiw6ZtRYZQ!RLlO'
            b'aQ&|a7m{NIn0foyoVt?<Eb2tvu;E38I>WnpxpBZ8l&A8^Bf~z%BNoc^_<zx#NDkVFK}_beH412}H|%#U)Rb*Bb0Dj0ahx(nEYqXf'
            b'1Tm1ZWkqJpKjo3_J_aOzW%J@0s4GeK3Ny995*pI7S|}o=B*&kClk#+xF*5T{X_}1)BB*QuUtVyfVW`$@zr0~+B`?0e+G-=zMnQSO'
            b'_B2mfZAs4mC~RfxeiPkpnD}vMOOn34JstUf5ldw)v2kz(3)r=3*4f?rA+p@G?$RvfE->YDNc0(CK1REn$moZ_RHDWZ_tLD?PJ<mv'
            b't#lh~zk7oZabM*oX=ftF<eL6#u<4^)^Y|PLcE6=eWPQLYEnWKvp>Fm0p2*A{%$3p^Ke8!r%lQ6bSMzOYsFxyIwiaO&E>;Uy>s0Z}'
            b'FXDAWys1A<eYiqR#ZO&inL(bs!Y5eoe}@%n$@IL<EDOLN*xphb{&{97<U4dxnxbBM^7ln6S=xlw_TCTSbqD@$@BQ8JxMTYYkGKkt'
            b'VY0pV3{--d-`;&bF8`;5y7RXr*2<?pg8q#ImA2&xJ*&2>ddqm*%II)5PVj$=v7RlUtLfpnV*CFyF%)pInG%eE!2d2WR9IcgYyrcI'
            b'1H7eUZ+Al)M151tmU7Z`vgD?dZ51)qS#>gG`;`OFX&NZoulsX6o&J1%%~^j@DaI<$w4>BPGaHUqZA7l|x?1pU6()R&R2{?qa6D{&'
            b'JnJ{%Un5&q1#D$>T&BX&%>@o71HLkklOB!mD_fXRaBtkg<D5_l8zv&B)B)^KO!S@^w*rhzB0M@wC{vgc?D?o17BKhLc`R2)S4JbC'
            b'jBGT@*$S_d<WwMU@SyFipKawI4b1Q3aTplAbdY8DprpGI+t&MHi~;(t_~^||M__B~_TEcosXVqo10K)8@<0zojKcu!YO?2D?%!I*'
            b'OSh&PYYpMnWD@o0Pe_KfM!lXhisdK3NC2`mj@1r>)gpkQrsC81=O>lvho^>j-kcH|IWHTAq3lPz<-zCS$HRe4=aPr%csP;fxwvXb'
            b'Kc*}I%ptJ&038k>EqTpIP8z0p{uSrVd3@+Wnb-M{cq}lP1j|g5i5gZNH()9Q!ijg{fLo?OLx2m)pB_po=E1x<G(~)$vDp-hRmR8S'
            b'g;NZbxHg-|W4!vxHy^wNcTJv&@Oby}YVM=kbjwZ@GlryMcG1)kEim3BNK+%%oYUd_R!(B_V4-|^q`MeSS(Uq}`-M~52t`i&mT*iw'
            b'JiGrD^C5dl5+`mAI2#s0SGWRs;$e!8nn`$RiWuvPOTIRTqA90ooR?+}rEhy>f+S!54m8u+Dxxfu9DMN?`RZS5b$dXQ4dlEE9Z#Od'
            b'5}`PiG3f6SoHE~{)e@9Ly`{ws$(4qNv|Hxdh7DzxS+Su7g*iIjfHUO7rIByv;&p0xi?Di$UqJiQ26&W`szJE2hqERhQRl7=k=;Kk'
            b'5LJU8%cpdiwyRNmY4yD-j>M931gZ*GQ#m}SdU{UBj-&Bvatoae4m)9Ow~Z%*GC!KK=9$wd6Gdr6=&`-8axm!5+lSsKj1zmNJh4*I'
            b'F}YJGIjNLtI_<^JauU4HEj+wQNvDRMF&NQLa~8+G;LG@&qd)aR6hvx3nFWt|&M1~_5OVfMiozT+*mt@BFcp9<+AmN>HdToLbdK1D'
            b'UPu8`;x?0Z3Om7w!#w~K3bUJ2*DY@mCY435HBKF<*U#3=pbIa;#CtnA`{B$gv8<u8s{v;b=nhP$cIA5NHXoV*UG6vZ5LXNxP2y4#'
            b'smo(-l@e7=H3nERbhi=#R^rN2#Xy<{xezc0Wf5S(Z?r0wG9FnrLmK`$P&|a+B?p-hUK;OSl_xT-4tNvI4!8Kc4Bx()tSkPNj1WBW'
            b'v#?>xmm0hTVrdi1b%kndWctG)ow5=ClQ@eo{cZ`(z!D%ZIsI51_P%`4xf<F8ZqPkA<MR7FybUZaWB&d*Rg9Xl#N&)`&18J7CCt#y'
            b'z-kkeOhB?CYz3`?1i;yii&84l%1fJJy5S-h8Y0Fp#()FBh|k3fisB8^@9>T;wiv8NX<P^S!nny8lCBNHa#<j%d{erMB40s@4e*9T'
            b'GzmCq<GqbnybxrJ7{y9Xa)w8y1{cV_60>xZ1jH95K(KKGHdQ>K+Y!Nho?2+6TzyQc#7iig&XLeX7ci}SJR6oYOMvSeUmhixaoR?!'
            b'OzoP^K^RO|^SN7=9%U7!1xHC#6y=veuMV%HIvH6PesTM4@k;bN`XcqFv_Zoc;W6?Hy(vjmuoA!t+(%&1eNVhKc?TTS&mT674p+JG'
            b'0&nL?U1TTMj#Le&Q--Q41z}%#n|>+BvX#rtK6%1;;l!0$nvFa;6H^drC>S!m@s{W=UhV*3S4_CZthpdg3)lJ0L}IFhqw|sZ(srck'
            b'XYeg+I<n346X>of-{A*J#VdRP<^6<|RSYIJtZ^-gY8B;arLdbu2C^PcQD@56wduY&mDVtB_De(Ek0)3$BAT^K-0RN|XnF&B#S1}c'
            b'?Yd#D8eY9Bw5T*#Q~jc<wG-FtOdl<juaBtRe5pS$9O@2RgQxR217B0+x?2joFrckSBSgf5cq#k``#ItP>m~VX>$%KB0}TXk{?ZlO'
            b'b@6-p{11#R{<=1r*3iV)2<!k+o{dZxz#CT3xgkK(%UcIc(Z<LaUpsi2cY2^s{22RK!)nA{B2Lg8ks$wG<drEd_2hOKOK!1#6{e&#'
            b'TemQ()T)zJf-bNzS_`}{*dbSMI9)_2W`~qnl{xw!F%3VQ03F}NelCY}ypx=Ew@8qtC^Ck$ACEmWPVfcuc^=%zpINeslgQZ(y^%Ku'
            b')WxZF03=5i1gE<H<Ql2$VC?3_20#$COy;8yCA*)ltJ$h+0C~xGyY{b;hH`<{N>>XPOGkCL4OWWQBa@x<=N7b-ddn@ych-EnVFo8P'
            b'lrKfQQHmBL;o$`owc|EzZ%|VkHrS{WA@}&{Tl?7V9Z?aHiBUV{Ihp*1ni*P!J){cVm%06TH#rX`7K&`}Uh_p&J)Gh7u~{by6+GVO'
            b'mjifo(ipm+M_#T&ji_Sj`j!P3YiKHXjGMHD1p2%{i4`0vRrWYUbBirFWwQc_HgE8@jJi{guwlh29dsq}OjWlO$J-?eNc-Q>xqfKH'
            b'kW6l&1lTCE6oq+DP7U579e>%r`i`DJ?T}!eu4})zeO1s6tF*{bEd8ubII8v<c~}Fu!idIlF6pH!@vmJq4=9U*{O$6{`4W6Z&|Ni*'
            b'DG^G-w}R;vbE}aWRmXy)RP2nu5#5j~gr&_ZAvUY(`#Xil>IPS}5YBHU?iT{QNwNcQ0JRuFR}PaN9rt@wg}|)XVBQ5&3D!Iqzn@<a'
            b'$HCY`;94%8Hjoki{E~MBpkg#w8l`417Dzb2eWHQ|W_~*P$SU@XrB(7GY=zLTF`T?yOIO<H3O(h4Gl+C^eH_s8z;B{p4&O*d;4{Ic'
            b'UeeeYF1-zxOHy1JyJAUz_jcL*M8|~h(bGLU^nnaR5<Onpd&9k<(mLJBo1ZO{PD<oqkc2@726LjxNgVE_L=~>?R5Bu4STqGV)-_Fw'
            b'Tz3|R$)|m6n4(e9uq;%LA(Z_ggxZK%NIONpP%vD|Lz{L<J`SlvJ;+AoXnHz1{7Y|w$xvP1t5z|VqZLJ0^?Xr0oi(IJn&th#AK2J&'
            b'6+u7h{=&F+Y!CG6Rkx~*k8AVSJLR`JcIWmJJLAb}-c??6%eOgIZLl!em{X&?|EjNpnaA<9eSEucOuynANBB>nMV?&mOW><oJF2x)'
            b'Xa&SV6Z~CxZY0%KC%&q4YUp5*qNE_CTZ{HKq)X!Km8qsOBVwOv)o%Mq=u5F2B1X9)?GkwjmM+I~$(_u*4I-?*5=Eu~zyr<9>XwTb'
            b'Z>k#gE!Bh2F4n{zF`j4?Msp|2r2Q(4aP>iJx!!;My7_vi(Rs7m?7V$be{VpT$b3T{S&5>;cw;UC6e{~tg2830G9`>v?G0sMj@N}@'
            b'%<2s!&bdTQu^=l2@KYN@`8|WG{0!H%9Qitadfh>n(q-q7zgMIv$TwfuFFu;5Vr9}-fKnLog}IeQIH|Fm`)30H-P`ZBiwP1sx;9z{'
            b'k<UI>frW800Frp1&?4C$Tjj?>4<P1nLLDyoe6fW}-c`N%C`+9@7Edz@V^Q?4wr?t{;u%#w@T1zR3JEGR7qvhkMws2lO^&VVqH@o?'
            b'KCEw-#E9Y%SR+#7({%fWA@`Vjb`n0AmjWAX-{^W!))|;-(dB6;rkh~Yrjn`?%V>@A$1VYv6H!he7Qy7ohB7yFw~$MO(rH0XYFHq|'
            b'P`6Yl3<x)$Rd=OQUPi1(z4}V%k1ypYFa>muR=DlDrUzBGYU=Oxf`zOuZ0DzSDR^%_`HAQ*o_#%)7rCuUJsy?pqn8WW3dK0Qtmg+d'
            b'Go=6-MfhaRgvdO`Y!%UbS^HMQkP@N6KKPO`a}&e1+f?4#5gv7{@l=O*wPEyW@Eo?Zc*aKkT{bJ{OYkBT5b$|#2TZi(+Q@l;twFqY'
            b'%~k)divwC|5lfElQ0L;=B(M;LC>aP^%O&@`!jg0axK)#Vz=-vCyZDc4#v^VV7*9KKi^}~m`tPv|j9+uNTFM1zoT<??DQucuPfI|g'
            b'MY#sg40m@rkG%{vR31~x38dh&mPC|EJGQ$@06r^T_*nHy8L*cUZDVZ(BkXi+7tE}xASp{MP&o#pTqKHm#pPx&M&0-@YNHut9CVFO'
            b'Sz$G=uV$2BFjI%P7Mjl)gCHHSCZsximOgg5j-#^E)Gh6NYw5POyuGd#A~W$e)NPc7bL|>}1zO3g{ZV~DN^PK;hv*+vbg7O)No`Oi'
            b'tp27NMYHAT8_OS%Qxvf=n7L7*1vr6`-bg_OOm5N`f2zQ>D^Hoj1MWgU=+>~}j*%Jgjol_C`E;oZK8|SeM05gSwUmQTRo)a`!O<E)'
            b'+gn%lQfE~+0p*0br|eF2%T|**;eVpdWaOf00`2PUZ5d_*%~-EgS#~O=W8d8j{rnWzC@1C?&WJBHI@ln*Ql~4FPr&cg3&|f;eJtG<'
            b'^DxbfGgm~`Wl@sG8D%=DMV5k>qdag)(^3(%5tdM?2wuwJkjTCznTub`{)2+AMW>&BMK8T%%EaKxYP8i=Rn}ZJ-et3`aAN>zA7GiA'
            b'X;yV}^OlANuwm{}#mG)2yhIN*K3RiLhS~Jd_&pQ0AXJ(NRi0{Me;f*fj#{8$DWC<qaWaF`4BkmeNyZAu1ItrX63fL%Ub%uG)(xz;'
            b's#m}QH1J1klEH3bZ<j&dGRn$5#?`7YwPmPmRZ*G}CGw3%Ud7HRZ-<(&D|?gz*YbI;-esv>ccV&|KYLO-DYA{r?+I*RbwgN}4O33T'
            b'>T|MyOhQ$l)m4N#*P&*Fv~Gx8bxLDPHTA<SPZ45RF&7LzJy18dxE|ZX4c0d)NrTo=LP$K=>)#{_`3Ebog`Vc)DPG?=V(!25UCdD~'
            b'DBT;3p_}f6@lDFk#s<++FQ?z2lvjII#orBK&sSueiga2{l@<+}R%v-r8xs9tZp#+U)pPY@T7PwuL_)c-f+4`-85UF1FzYi!kwsBE'
            b'vx?6wpeo4PQB=utmS6td8lUP<uirOoTdF^UVZS##{EP=|C+F2o6>rIes=Sb9p+T3_G!Cm}SUL1pl+CK7B$8p#^}tvb*k%~A(qKPw'
            b'9QSSIfy77^$CGd@o}bfPa_jJdlDRg_s<823qa|O-W*FqJF%SS?U(r{WHU=}jBQF2ktxQyVWK>Nn#$oM-WVcP3K_~)!zvX4F8ce31'
            b'H}c)s%28M^2{W!uAO1Ng8e}GDocth>@C#bAy8YEF)9mIh4Vmp=Fct8EZP(@CvgAvOwp(R#Acq$UEiY;OhDn_!37biOEOS)yT=R$i'
            b'<N``2cP0TcB!N}9yH6G;(frkwS@ne&xT__hJx={e3g-z%wTf&I0Oc%6TQ}3(B)7@RVW{0Zih$5Kbab!7v^n}q6@qvus}7^|#g&Cd'
            b'l?!L9MEy?1lj|^A<!`w-VP1n!U!yWZ>U_Kc^7Lo$TJbms3|d_V&G20d7dY|;RU<>437KuW?gW+7F&pIhkuD-q(WXzOjJ9MIRlZZM'
            b'N>T}7)ogNIsxrogrk(9~dzC9IzsCVotlw5up00TLQCF#Zq>_#?Z=vK&qqL8^`|{4<17{fHftWbL-EF>UZ@1p=h!2=M+T$nh-n4c)'
            b';zLWww>wsXxvm?l6KvG|N;E>L)~mXQc-}wzr{3YPcO*LRgi2~N2LsbFjNa2U$)qDrdIy7xK6HQGdEMF-lrlhf8HYO}oqLm8mD_`x'
            b'U}9%e6`F?gNU@+Y0@aCR)oZ$9%tpLdQp?7gFh-|};!Z}ZT|&xeunT<IWh(7Ujr-rd@2{#`npcb9HF=U;Gj3hrVEJ{=T2YolvvJF_'
            b'ru8H|A0{jHnBLHjs9NPv)n?TajwsAa7V?-B^L)0p{_EfWLs9J3ma+>e$Ao^FlYeZ}aSrquYev^_qd32+DRZjVoNTG*q${~t^^#7H'
            b'DKi^?)rF27GFi!<AIn<0=^7+r5=ZKXeeS)GZ%D>6C!0?<c)&y^U~>^!Fnt{<-BB9T&!hgvzmD2J`${R`n9hwq81?0#|EV_^3eY`t'
            b'5u*LBDN}zqnMgAvVn%XqL%e+>J~-nb%Yvj~)Oo}~<-AnrQCOFot(wsMV}4$e2F)>W9^#<g*xl~5h?VhJpT=!B_Flhhy?HBA>jxL|'
            b'-~CXzNgi~43p<o@>^9Au@yf^F>-n#L`%lr{+irE3-&rQ`kA&b3;ZMz|!pZ7aEwIWZ*DW0c3El&PZJ{40oyu~Xgc>Ys8UINhu+xQX'
            b'92xg`YYTsqG?hQh0jr=j;T@QFN<43EwVdHSkjuImVNFf0mzpYr!qaeC*Fg~N=6zKTPfNL{elvq^$}Gdltt_dkP-Ue}t|=qSOY==h'
            b'X{i`};Va^lgce5AAW78kSLmOC<>(BSAw_~s>BmkD9>!nH$nYn=Q~_7@dt9X-?#-t^?R`2J9{<`STPA!!T&Em-QfZl~a9_S_y=${B'
            b'dD#ZK*xeH!>hC4--QvAud2>Min5~-EbM!n>Fyl+dGTwh5C$|(E!Q$e`zxu~7__qTWF`eAvRe8K-AMpKXRZLwKEijYF+)@dHyi~za'
            b'5-*z}U)R-#7KL9d7~OSR)_{0(NpclSlUDV8dzrANuh&=j#+xq9Zf)6CT;2I^0&&n`'
        ),
    ),
    'fx2_rc2.py': (
        '3cbddcf85e82d7a17e3f19e649a8af1901ea62fd5d91a7ca0d13f1f7edbcec79',
        (
            b'c-q~4-E!MVvgSRWqJ|wkP0A!K%JQH3uw!M3wiRPbUWsl`heAt%1SsK%1ZV)1#F;Y@d%4#e`-JC7_RGwwLIEIU**mkbv5IbsL=~#C'
            b'vhu&OQjJC<2<D^d{f_FW{5n*NG#UHjXcpzG&c~C(!3X8vN7<fAqseucs)@hM{F%B6lX;k@EA@n5Iy)OY-T(AaWx0P9c0KRQ`~9IB'
            b'zCSus`-g{TYIve9FJ+qE<z-v#A0MhO@BdJP{e$<^<7jwva;(t%?df2E*PYG;&uthy0Gh&ov*<S5Q@)x;_hHakBvG8JyD+-C&cnca'
            b'H#qq?7@i&-sCnqeDw(Pz2r@Na&hlt6i^4Qhd6L|y&hr=B-R+m^4Z2Ku!uZAZHrj5&7$b+XK>4}yy>XOxq9Ba(XaY<G;UozNz@l4M'
            b'siQno$z6<P9}oUI938zKoes`Ee;gcQMX^7p{&$JDjDfHeFsdjI=UH3%F|8B}nkT_>7ShtjNq$W*1D?5B%$6BLHc8?<yw5v9ludvi'
            b'Yy_dgqo!mLrhcBJ%Fk{v@-#^S&w%^P!^yQDN7-CqD|z&*Fb=V^Dh($gaESfIV(`rmqWH>7;@L{=e>ffNAO6uf92@}Whbmjm$H@%9'
            b'W{L7=GlDJ3X{;0wcY3{7YMO@FjlYFyqVm*_vw4)$K8#m+h}Yso%|bto2?zcdi_EU0#Uc#2<raudIsss!O~)#Mpg0Ff^tzy@fjZd#'
            b'e765V4F?}Teb^rk)R+A;_4)X4aN7BN2H2X|;cS`8VA!iHTLRZDm!!I*Y~kmTKO4=$+i-^cpjki*g4hp(w?p2B&!5x+)HK6B0t-Ol'
            b'@EW^+=Vzb*dLvA4G0ZKt;yQ^Yp_=<Z++CDkC(B$37(gM9DI}tj6zk%F?3%qh|2CYa*i1rT@AbCjuQXXMG8GUWVl`fYkoeIkii7aJ'
            b'yL?4#L}2<Vh_HD%Ge_t#p#aWl1jIJYqIa;$(_V{qnTSPJO53Hvdw-J8R(qbO@WWFy@7rqCQhn9DSASM7p0&W^J8DZcSMt^K0PtMh'
            b'T|*!UEi%YpN+Jb6reD)61PsY+mfU3oy`KUx5Ew*heok@(P|~`*Ymh5&vae!d*Kx8;2^0Q|NN{DDOM{6?fsF`T#gHmS2KCQ!5HUJ<'
            b'*&HOvSbm$N4Bh8XoztUt?}yF=mzT}^Cq1mP(`#K`>b1*a)t~zG?;dc$82GcSzehOOYOeTe2k1bfx@s=yOAjyq8-H=f>AU0TpY7Tx'
            b'?l>==+i^UwmhhA&2U>SX?0I>)vT$x=S34cO5$HSk>;A#;!yiUI1W_I(G4a1AcqlL+q+^f<N$KC1EvdcC907cO)SLx&;PJ3QGsq)9'
            b'2CHK$Ucc(Tc)k1l=Qh4{Qj)yRqggnWug|u7KeNDyJAfk&GH;P&JXMA%wzh~h?EA%rG1txR{1txTzgs_=vE;y)<)XW_B|LR^oy<aS'
            b'3BF482w#_e00C5xA_1O3{9?qFYGNjV>}B`aF2HO)*=cunTKxUh&m=X@5`G2387s|+zajTYU0_7OWg#T!50aD^F%C2EZ4zcIfzmL`'
            b'fiY%Eui%0`6|z7kDi`{VAW5%$QUzcaVNWO+KzezZg|lh*CWbn}hR-3@+lC)JAi&j?G63-hw2KI62a7BiFxm0RP_4k9hPg<Wq|tZ@'
            b'>FfdKYZ6;7m)dq}xbo96>3Bi8!`I<GP!s_<ch|B2!WBr*pbit?B`I_nfj3FA{8#Qp>)|#6ZVo!ki1IUkN@m1PSA94+fV_Wy^6})|'
            b';COH}{DaJdef8!9Qh)#S4|O;g4i1JVr>q^sC^$Si<TbwE|DB8pQGFrQBQHQpe>DyL9P~_DGE<EpQS+6vpUS_25@UJJV17Cs93CBz'
            b'nbPQb$4UH{_t?5Q0bHjIMpu#%!k`YK=@jZW&b_ZokOvXJ(fpv!kho9L43W^QHno7_gm^7yv|r8rdu@|I?Jpvz@4F}qt;W=wZ@7zi'
            b'BMa@0yYK>_WelTWp1`!oXmg_2^VK8-LBfoJCMHvYHO;q2gE=Ay;DOMSbY&3DJ3SeYnch#WI^~TgQ-nz&%CeAXBc1>`pg^Ls%gd1+'
            b'e*|qYPsn6}vSGyy9uXuR9-R%3jt_=vGV^&9f1YUVTZjuSObd8DS{F&#WWHDu>yS~Gh7gIY;>-epQ4lahEzxgsmBbh%PqaxI1`wh^'
            b'3U8rDrVnWDnOG|@i`7#_hVdDUCm6kF!~J&y)np+{fMQeGWahEJ!)m;Yu%$s1u=KcwbwFb1V{}iX3oyqK<O|G;`8Z^ubChFMG4+M%'
            b'i!B;QekP0#VSNkZA=9Y6-%}AAksdG+vrYx+&1qp7$<^)d>zCMXkj<QA`xS{ELz8*B%vr;r({p-~hGbuPX(%=}!6p`}fea%o{3s<}'
            b'ZmV%fh!3^MlOC>;&p{bvZCc@+!FpY33G-?HzXFczPQPOKL$JpBzYv*NWtjewsgp0q-pS|Tr_V#$v@?DKM5oT!a#aIaI1w_MA>jPL'
            b'U$7}R_ZPDB4Hic{CMaO$>1_^!25r2wLIN*gY^<<LP&qJXAsM`(){Ni-o`8uv4VRhT2yORarU~RM;X|wX7(JQ!@IOm;(jzcUw)iz{'
            b'J1t*ieVXoGUSgSm^lu@#7(4&1*|~YT16u~2p>d!GjcpCk;Ng&W;t?<PU}t<C0Z*5g4hw>_NfHtlK%i#;d*%la-fd;=5C%!4RFq>V'
            b'NYRO2(wKz)EV&Y0Kt2(05{V^|gZ0{lg(7&1nIrCe;$7kyMNM&%&iTz`jF-qc;EjOx%hNEQTw`s0nx#X&8({xJf>E}JU?CFJ#y7^J'
            b'rxv2F#^FTtGwq6&tY`Zl2kI~E*xqJ0tIjY@rjUcTB3MWVMoEdPv5jpE4^>!b27fYH!j)vr3mu3VWaq~r!E|}4(OuH0mzRqX1UNRq'
            b'V-{wXf~_T<7aKXW%w{qsHv<D-M+{Ebf;iz8_GB$ps;@1LWsGP2+JNM@EPx$euxJ4eP=MDb&>6Fpw&G{eIF)5+$w2gcmq1>M=7fg<'
            b'(<+a`LS3y`e;9`}g<D8o_=M<>WS=J6aR}oJ9@xjD<Iw>W_Ym_lWw}ncYr*!q+cJf2UeWYmay%z>S8~9AHOeVWry-lO&PoV?Yhcsh'
            b'PlYfES$H!7fe!dlp3o=zTG};|olkf|ynY7>#rEkwk-&^BJX)RzA+lvM>!Fa>O8^2#1SU4MgWoVwA!;Hktb_0>C4Z47i((MaI3%6T'
            b'{x-A@#9>I&Ux%{=Het6*iepGBj*KB_lHuU5L(+K%C%+F)|Il)W#h-}`fQSN;@+^Sv=pRuianQMf)scmP2-uZ0p6(z21^)9CGMgke'
            b'EKA@F_Bo07PO>0+AaelB2!}e(Ma5`m8%rfW!KcMRa}k;$NAgItq7X+AdQdx8;29Ep;Vla%D5MZy#WY|ruGrlr!Wfg`9ISj0xQ^s#'
            b'Va@iv$A*MrQ-npzif~6*lh6XisbcI2tO&TqEJ+w*2QJb10!QqTy|sWswjMmp@0h602Qf^HxocmS(e6m{i7WviDMdN7#P>7LL{`X#'
            b'y3&Ca4094|emfhUem)p}J{_n@ny|^D!6d!w&sJHKX(Pk4!I`GdJ2+iR)I~<lMA(uS0h?wLvhUyem7*!UJdUfdMU_Mx7xBtLmRo<e'
            b'6l1f9)v)B(oFEiV9f-D#+TaPni$Mq)i9777MDdiQxFCoIgPCiHBoHqM(i(D@oQ_$v5YLAhm3;@+8=e-piS=?+#mmXG6n|-6j&dfz'
            b'NXOiWL%9bT|0Ge42WMx1`*Tduud70}*dg}_MVFUQ85CBzyo4N=4|uX;AJHdBu|;wRAQJM$x3=h0Y=S&dMqt6exE9|ATbPr`CsMxi'
            b'XE&J}+k*n%KO3?I3fa#F0*KIt>V$hGmZw!#K}kkWx1DF?F2+}FPdHI@9dMTtqhO@V%O}wNThJdV1xK^5X-?|LBg6URlpGsOO^yw5'
            b'zqk?UQn2U{trLNTjc}&~p&2jNTN4Ro>F-KwCahqTg;y-LgVBOK2`v?&dZBX~jfM^}jYiWY`Dmk&islO#--`P&LNd?O&*J5L!88&t'
            b'^m|=w>Ihp$9^YidVsWG~LS!PZH91=I@#wFE(~*{_XKjA=WkB(w5#ePpIyoN5)3+2_{zQ@G4|?Q}G_W-C$G84Gnyuu4NG`1I&UE38'
            b'7{UIV=JG%rHX{Oglu~z~$7|^&z{nesGqDNq(b?zIPp3y`gVFxkfedJT;?ak}?}HC|F~^@jjt)NTLtW_yCx?e;qxV>$^Yz2Y7y0(-'
            b'{XP~BC8HPp>Gb5yXz)qDdiwdeH1_Awo1<g>`5gu1&WIl$zlOJHJFrdI!85UrOV4RIZ#05!2r}rabAGeo1Ozll4S}O^_WtP8r@<lZ'
            b'>D!}s_UR|+&;1X()y}}5*?jY?G`oFxzw?nif&Kd^<IOe=)BBy#lsA5q!T8JE=Z5k`A*1EO^WHOYVIj|6?RsC14u|jY`FU^0^L~PY'
            b'>)6N%S-{#~wIP%y#c(^1{P8@Am5)x|zC9ZZ&-PTlT+G6A`8OhS1OHsKRkz!{pjc<K(`&137?MpGU!5L(=$Q{4`P3rjYzFsj(09xA'
            b'`*jGtaK+3LD$$k<66b`%;L_9T1}B6p(v+i%^_X`#)aZRH)X2feBE+}zyc~Q~Zt!t`HuE^B=W8>Ze5qu4C073BWjO$+)gy=W$xI+G'
            b'3I@k?8cm`wo~%4pO>?&D_$0!JWDXCKpcgq;p-b$;3u?H$+-8S|G<PLvL7}@}RYEQUM|{Ysd!m}x$sIuB;Cngh;nkuZ7&1*)Zuo-r'
            b'F|3~!>AH+|%4drodj5Hr;jHr;k?nVa%13HTcK-pT4cTbD;eA)=1_qqZo-|<G<v3AKyc!8zc_ws4y8E}j+A)oKvW}=H8;{~eIxJX3'
            b'Zp6rMp6)!J^$Hdw`9d-h_z_`A7<)rzqluU3kOkQ-Fgsz9x;mCTg0)V+S91ErTX%eY<LR@P4|UaHcRes(Ik<Jo8kzlb&zQV^;WL+f'
            b'b~0#RZvQMG%VA6C&wpXfE=R9|K3ACTDWa=$bZsj~;w0mL2M=73y!pGx$z(ch?6Jrp<D;de=T|sx^w{7JeC^N2fv=`}YI-is7pC)M'
            b'mW|8{j=F=fv*C5T(714#!9TS@$57U0P|qCTR5YwRgY8qA$<Wuv>Eu0kC2J=xFYEijXgVvdzWJmKWb4sTav=6waIO&=j|X+l%M20E'
            b'{0LD0L9F(LJ;x56Mv7OEK>SVh$)FobWC;RV1UFF(8zP6#A#6OD;#5)QMmqK+1Z3V~&R_|q=F5x@e!$1UauSwY7bhg3MNSY-Rn>-Q'
            b'Le%_LWuS{a!9Gb`JK!%S77SIE%!>2?6R>hsu{9%>@hq9#kSeAD$*<J`AF|Kl4Bcy?+NE`!w0z_^?O%qaNErkk+qH>yn&HFecVWmf'
            b'Co7`KI@C!)Pz|L>?HNU6*jEzS$b)Hl0wv1FpVV@f7`KCDKOGhrpOlgzD3~N?C;CJwu)%d6P&|Ip?-C#7fToUynP3|PUe_R-e?Q)<'
            b'aJNzL%O4@{XgizbU2aJaA^~kD?T%m%E#TXQ&1q}TatIuSWg3^*qiQ%X3z$)aJw7|kv1I00GoNApf5Lp<5H7{!B0L_u(nkn_hgPa5'
            b'(~$lXCi?~^sxun|tY7an8Jhj?1>swmt}Kuzc_clMlt@y4_AMxPWE?*2>j_luqn(XVJPLixAl>DHf{BH8`MdqZ%vs}^H3mRsGJ}M6'
            b'o#9&T4WnDUx{7YS!mCfW)+cv=YSWavRG})fPJO{^Rjyfc`Q;5mo1OUfYS*jxG79PoE>H8A)z&!ZM`5emO`j{KNEyi5#1BJT<DvTY'
            b'?CAfiSgP~i#=#9Nuo$v@=lP2tBFjzduFX>Kf+@d;M85;fhiG>b8T}BL{}3C|bA*#t*w!2?hi=vyWMz+LgLQVd&H%M$T0)2Ur_W$5'
            b'#W(J89ZaF1Ym5(jprdVwa>%k(>hmp;S(xsm0ABs*ro5`-`-hGASGA#@sCdzxM{zV?&YM<Oso#E6&)Vu`>tX7n6<R8O>>}$7^5_*l'
            b'!Y=w3tk6oP@7v6}0Q`aNt+mn5cRHEhpo`iRt=f}+En3-zVYK#M{19Gu@qh2dzZ{RdWh?9fBkUne_FjAkDtFqd_x$^D`F~2NyZ=mL'
            b'UHkM$(7$n@+O|BR=eq4$zh$y*b#!<eC-{GhvA$bCucwFa72E$m6GH_Tn<>F@wf^rC!y2p0m@P28By?6gW@~OZc5Q8nhH(yzP0reM'
            b'l4fF2Hj76G!@fvJFUvw>`}KcGrqf?8FD1(;u0^jDnRcA2<YwQ<s-2i1U5^RBE~8Z9^XsF7Z;pnYw<o6^{C2YQOejpp9d+srXG%+$'
            b'4*1G_NwRVRTVi3x;hl5qM{>|KY?SzzaR=C=lvsM@WFb453v?)xa%LhY^vcm37BKTyMU>125=J9XMm`!9Y(?qo1r_KU17thfXIsTb'
            b'2lIP$&;X;Cj|j>=ILT)uw)LJGQ|Pm=-ug4w5!l+gx%1PjOb3L?fTt6NBB;T!xF{sMn(g_f^zSZ`MYCZV8*SBW=%mfjzY`fY+O1Z>'
            b'C?!7uBLU>nIHh(B)`$T^!$kh*=O>d{MCUQL{)`i-Bo7qBaJCg?l~Zbcu8xLhI(?+Wx0g8Vn-#f#X2>AnQ%5?pEb#$393U-ytwB#3'
            b'WktS+<mvblPUcL{Vn_<OPA1_Zw`5|5HGy0@Iwd&i4i30w3K{|~IA3$9saOW{XK0G${#2)<=F40T94ql*lPGgFOU887PBN|ily*%='
            b'n`OMaWH}4SZMqRBN*F^^vAAgF@C%GL3A4<}ZI$$!pj(ib4%yb@^0tfPl&y0YZNF&B8)3+4uOu8(pH4-8r~IoynkH$p0nWxE_{vDA'
            b'W67@QXt;!zWrVT5I+trYB$^7Erg?ehNc*-IE=c<2Z$L9`EMq?PF2GlR*RTGg(JBw<vVor0pyScgltd^^<qZ0#l%_1UXt{uLXmzc)'
            b'VY$+_koN0b+b%=d7glUrLE(;0H*OsHaPH*yH0k;#-eRnt(-&xe-T<Fd;4BCi<>5+`kGONc3{km%Tp_BqJl0S7@?PJf_`KBjqB@dD'
            b'$`Mo*t!C};Q1$$r4|GQ3<>UsP9S-{_+HRZ9Ty#DrXXy&3@fp9-i0HAsXHK{Ij?J6FJBmE}t~@EFzEU!oO{y>{oP3T(pY<eke^mK&'
            b'bBoVod}lDGpUzd11WGRdN>1?9k4O-i0d>|b<yDecDuYn4KT#CrtYF_c;drWmF5WLFqcT;g5IRR~qZd)Yl)25NoytxK;%E<GLQytl'
            b')@;_dh>}?1W@(%js5i)0i?ELuG4YGtg8gV_lgQFG*_^;x4BbKLn@zo*R+$f7fX?^Yc8Cjsjwfky(ppo-+^QwUxoQkpa&)&50i~3H'
            b'$BKb03kxCO3@Rd^1i#a2Sju@+vKg}IZ=vBK`mQ<1h49>Y_o6<LYjxmFGCP{p=XLn@+~h#@A8drsSxN~Tu6$|WC5WX>Ft-e<sZr>U'
            b'hV<%2{7>p6rW?KsG=n7|FeTkl9SlBvu&3bG1a9-a8|Sj^EV>CxTqgYeOJ*1~6Nx7osFukjjXuiB&Y;`vd`!!-BI<_SiUgq9or@VJ'
            b'g~iW0QMTdY2pJ;IFwTHyz(^pMrK{qN5}W8wAhkHG#tAa}#lob?8IrFlVYwm@O@15azUx<rVnfQPjwhiUhyPl}D_uZvMvN09C*|xs'
            b'H-jr;UyWJLT!;ALqh&E}U{fU%&c+UBv#f+h&JCuU(3JEp2jwXz*2A<4=+sTqECX(DnE0H!BM0K$I<;FmhcK8fXR~Hqdel{v6&y8D'
            b'QI%f~y?U~a^~sO6&_wgJr=F_Q9eZ(eQ$m*Gi^v$o1;UJ_Dy#%JfqMijx$mjJV(&nL`o+Vx)8V2JUhsAf&Ba-E?O4@FqFQ8b!87)a'
            b'w;9y3*Sdw=JY`Rq65N|Q|E*o5y-{)(4}~GKYkxuR;`uHByAZ;4W-SD9R=GaxCN6U24V(WcmqcT8Utex@a>`|K^2>MI<jycCm97kh'
            b'oKX>R79&h-tZ}7@Y8e*^d)Q4U<I#?1s56u6p?trUukt%LiM62~Bon%!9$$4`+}qFhWO@U=>Sx7iw6<Yo8lFC_w3q}#SN)QzEhoyh'
            b'gg&|?UmsAr#ZrG@IMN+u4W7-C9KNQ>tt~0=%7A4>8WSS!)e{xmm!A_Js6I=uDm~X3KF~mT^S8d*ZmB=>=YJDy$=8+Bv_g}lEtdmC'
            b'MV=>PfNpf4b6Y{u>n!Jv>QQ7&(wv_Zo$k4lJf=RSVJ%|M2`6NZXpsM`iktA~cJgu<OK-7#6;pBoW3w`<)~b_bN-nT7S{L3IcBpAL'
            b'oD-j`xeRqy6^{Ndl=d4<K*!ffP{<)0?_{UFStUqU6gfkdACG-9PUr>qc@f+wo|R;kr00qodaW~+%|(q>2$Ew9LXwDoYC1{6Fm{W}'
            b'%@9Ogm-#qEDeh;pY1yjp0C_HVN%k&?h6;h!&6e{fm8$7)JFFD1MlL(q&t0_Cdh0DHcGi8n?FJ_;luuNzU5geI;Zef7+3_-MZ%|Vk'
            b'HrTnBuJ`!yTl>)N9dglviBqfLdvc^5H8U&~63{AiPiIZjUC1JsSh>_Mdo35^>~N0PCuW_hR1EmFNV}HNS!37&0!4WMGop#5+x&I8'
            b'9NsbTI5+nysmVn_gi>&%R!QK1%q_9tjLixX?OaoKzPZCouu+8^8+2vy%vF92$Cs}}v-ZE`bN#3kLvp!=Qw!s~q$u2bIcD%K>-h8D'
            b'#W(Z>X@`_DAe-eEw=XKXVVPAqN~B+@6OC%UMj6%tt}>#toO6EZLj6bI%mZank-vQzxmZGA2)b{EaV0`c_^x4k!`x=1_PS$1Rw|Y~'
            b'<rWFw$f$&+-DLYRtJ<3pmB;4B$a*1M+=AMx1bEYQ7jS@DoS>_ONe_=s2Yl<<y&>-2Ju#`PGMKzyTu7zC#6ytU5xzB$3I5_zX$(*)'
            b'8mx^{Hy8^N4r!nGg0!3Y%RaJ+JyR)%qUcj4^y>_#FZ1yg8osj28MXo<-;^DPygc|#9L$j$kOX|Lq_jkqIKyQ((PF`hD;HNR4e6es'
            b'n4jdB$h}>@*F!##V@Q(6%X{y*H(bi2zxJkC*QJvhc{n8Dkb%RTWO6cx`x#S3({`#EkzFd%0FG_CrbVGUE5j7iJ~T|#sA!igREZ(f'
            b'{UL(diCJhn#lBE6T<Sx+c1bY~t3x};M&;;wItBa-e?rMnec7vSHI`!)Mc?%Nxq3S5NR2En`oSOA*l`)7A9w%Rx#n0N=;_n`x;8nk'
            b'EneTLzqM2D+<RnaI$15c>I+20HfN>{3)77SHR}6z_M(k@9N*ZZw=2i=OS#KIzbY;I<a$p7zpi!1wC+?|fmk%*-^z0*sdjzht2w7e'
            b'hY~4j3L?5K(cXr1NqT*4s;SP1*b`c<o>JQo#|YkUAck%Qj|2f;XFw_~8u!$yKUR7`QbiN)a4v^DU6Qm-(_79H%<)e(q+A(`Lokg!'
            b'S7KBT3W|Xr6d~)7VA3Or6)S<7{4VK8fXh}A`p(;NMn6-DsQ79Gm7v8gPq*i2(}!&2kI}t*=cO%5#ufo79YnZlwK)5CZ8fJLiI!hf'
            b'J7rWnZZdF%tSbnlL@))%^YOXu7D-(=4%rr@Mz_Kpna9;(;5Zkv+OAx`#c{aJt522v_(YF_DbP7y(ze@+++5P6W52g5W{SCwSR9#T'
            b'FrfoB*2!i+`Eso9DVEAJ_*|!nU#=V(SA(Ith!YGK&ckpjFX@Z2v3r)(t)lt7@wGvIJ)@y~`k(`sE{4mtYkg}+6mMQpXkB)-?euBO'
            b'9Oc-=cWl(ZWU~ssgehl$z~>jc(36XmbMTC<A-wk8RsX3^1D4LF7vf@b=js6wScv090tBz+TzXzm9ij@{rm1|&Lq!++^vg8UIU@~B'
            b'M^3aw_5Kus^~DdNuZ3@;<pLV7%xIPrHqCCQWgxPuT*EWR=a|kCKYzC4cn%JMR2+wzh;nJC>@zTc-xaiZsAQRrnrex*@#?VS`yZ97'
            b'q^&C(rf(4QZDXfK7>Rnr<*r9;((avtIL<9)Ti8vP)hmkXI5&*l{0%996G++?qyrTKGbe!B(Ww{jGL_WKCB$ND`7)xuoM{%K)1fvL'
            b'>(bQ{%Y~yVv{ICK;-XM|zRZQL$Vgk$Wql|>YlC&d>Yu7nEL)Dgiu{3`Vz=30=JpybAX#MmMg|oyxy}-*Pu(n+8*zsR?%)dc8&qb&'
            b'IiCG0wyTkR_9nN)ggY`3hk;8i^$5T^la}v6SdC!qt&3Kzv$+;ba>CtH@fG`ZtH~U*KGJ4#5`rv&_U-m|9kYRFY^_yU<$Si12)jz7'
            b'pg4+iYREJz2X*Hb9byn(m=HG~>dHH<$_YG|oXPgoEXs1{;ERxTUX`Rt&gmv*k+lfYsE8i%v|RaSL=^(oM3D4wL}Xvnyh&f{xu*uc'
            b'E+0V!YkJuw^KrP8)8VaNS6ZiO#yjuy3~n4C%cltLX1b+(nng?70@!wUY0b#qT6mcr8gjgWPmbC2)+u5y7KAyg=W-(*>Q6&a<i`p$'
            b's&oMjF}c=hbZ!JiOG-9YAP?N)PF7+yDV$O-8%4#UTHW<45CIwnBQeRao7B50Aa6OfgdXB*Q;oD_sBEpHG$Ts%8||WWkyF!%G+|%&'
            b's6`L;^FqDr`ijksDqYvQuZ8Q&#^tvJHdx&e)^)>Fa$!#8#8zhl8qiuMOjzhpH%!%XM6L;ym!+Di*8dnGMr9IV@Y#X-g~j#I9&WI{'
            b'SxGvyP7*@np}hW0qEM=oQwu*W#xuOWam2!ZmkV*DLQpoZDUj*96V5j+JKGyXORJs(K~moA)jIxetMYtR##xioXsNP9gRWItU(|+d'
            b'Ys_7;MGN)3{xPqAeUnB)y>Sggz~UKJ^Mfe$E+UbY<>8`Kq~a>%(wEjzG#N`Jzx;a(yP2QC>8V@L%>0}UPY1(;_jJnf@zeUIHE-#x'
            b'mZExErNP!wb50C(Y%dCyoKk3{B#~j&^#D_mZAM4t7VHPkM7}P4NQ}&IJPXI_AtukIw~lU!xND=M?J^$9Xem~*83tW~n*o4*#b2Ae'
            b'F_dl_OU=*z+H9c*Mzyr!oS1FMINFq+K_V~+x_;g?gX#R=cCi~<1qxd=VI~EMqrZh!gUkg@PM|dsej{tvwm*I9n%&Z+t&f}?rUEaN'
            b'?YaV7k$h>@c3YdFCE<li%M%{I?NX;p!fxi3$Q+Zw)cK*GLO|&ZyEG(*WUwk1sM+FVn!mVk>v1px_pKx>kJEaT!g+*IZ6X^CKs{Bh'
            b'tebgmmfLLQ2-MA6j)3que12-fv<3QW1u$f&QkgOBiyI4#3v*p9Q&XJ$(PhR;H4mlsR#9Th-eqyG!2P83y`x{?wNgwS2Cc2~<M^(X'
            b'Ec&9s6e=)*dAH48{-=V*r3M|ux0M{$wArHpr!8H^Yu~AthFA;Xy4m!)OnHJ0O}pE#U#wkO{XGewQhB0v)t9O_&}<Q=2g;c^^H%D9'
            b'lnZi|FB$E9^Zu==<K?FM<28M<({Qq&X|Uq@RinyQXm)O0HLS?Q^I^I)kNFMzi0g6<O%*yz?3^Fsr*mB<fzI8xw*K>f{Lh!Y?e42x'
            b'^=50!m<W94%YT_m{@9l1?Q7E)P7+^>i<9E&l+JNnN%E7Olds58eJ}pYOq?l6RoyuPo$E@r@>o|5%vNBIG>J`N|2sdD8%v4Kk(5BU'
            b'44CSKNGX>GE6CG~PK?Qa9_<hL>u9GZUpOU`(ktl?Mts;meK$B8Du^LYOzyqz=%aW!+_wcY2{W%>c6WEwo3?uOQoZrUVV;L++o@<p'
            b'gX(#?q?IaNHnc+(f6T{Ev#>MvCpUzHUi<m>PIu?E%J`U#$L+OWJbT@J`ATJ_g3CqSZ<8zFqw6c|P$pr#EceFCfa>-C=YRiSs`p~M'
            b'yQBEXevW;Lbdhk{Au2U*N<^Bf)>O|+Dzu%2Dc#G!w(uV(o9e=1j2bH7mC!w}%v_M9k!g>&w&-t?Wx6moR>5ncob=a*Q@6Id-tZ3O'
            b'vg*WCJxnUYWk$F77S2+7Hlf|UKPHh%p&ifE%;wDXMKrn5#p4XBN@+|h&XDl4VpB3+DmmhEWs{RaqIepnsi|LW{|uZs>Qr!z2sYzC'
            b'cINOfRm{%mPkfkCb>{DJ)c$Z^G5zu2-Tv_C_W`T5=$>$$NgTqYNoA_}<aPIT4=4Z&Jn4Zhp1)9UTEA-IyP<o6I+IubF*`G_@7a0c'
            b'@JcQL>d^09lHRbJ3@ORgQ*H5U`P)JBgb(rP>MC7(4duR~DW+;J5xYqp&DxdX;_C2Wnk+hzT%WV?>q^Dq(d|yRG$7q1&|D?bq`Ut9'
            b't4<)X*In1RNt-UQZEckmddvIYwdSDE'
        ),
    ),
    'jg2_tail_reencode.py': (
        '3a89c2b2e64e5ec018e134179d8da57c00557a043c1a9d27c114bbcc67e5f9b8',
        (
            b'c-q~4YjfL1vgmjI3M^8$3_2u5%JM@@xK*@F$2yTE6(uK|)m1?t2}+1afD3?<6|MgJ?bi%u0FZWa)~D{hTf1u!81(e?^z>_bdcOPH'
            b'&N@qXCQ-Z-#&^!@A-_rD7Y)~SXS1dM`}MBVZad@A@W=_$=}mMOI+KSy%$#|WIzjA&vnUT|PM+L`u``dt#jNEdv6J6~&e?~<)6>yA'
            b'XOhewI*rD!AAWPrKOCMpqkkNpou73a=jw{P=v`eot2CMdoWRM#>vkHhf+!WB*n<<~&hzKr?{!}H-Z+ye&zx15Hl{a0e62cv{v6Pm'
            b'IdQPWCpYjxyk1Vi^jQYK(lB)H!{9a!vkY(t0O4ZpcnD}2E|ZpX)|)k^!Ce@@lNarm%|4A_kz8YwDZq5*X|e?L-W|SwKN^pY&uLhW'
            b'ceCuZoZg$=*6Wx1&fBJQf0JaPvrJ~;q7h}z+oMm%=c6$q#6(GA#Kg&NqE(9?z(CsB!*Y@=9QY;=oaj1E(lDc!vv3NKU<{3E66fLP'
            b'9B|2|!7OaiV6jOaOctRtP0}=+<`IzoKFFM88Ral7K;z+JA>&`Ivm5#%gW)vpg9n^g{GEoug6Z!7#;Y`$0LTaD!{JXKMq>bR4sBpq'
            b'&`Z19>GryN&Kb7)UHSuD^Q=L04n)WC(dZX24S;l(2XRh705-f>FK|?gFqaWhTN=zV46-!<6IK~IM<)lsb*JOQgOPJEJU%`-cizHZ'
            b'C*$$x;M@V+MgVsl222N}=nDE2k~O@mt8^Xb(K6ftpwVm{EPUbM&g!Ax=XO2}WVWV(_{2up)m6*szv_0|ulDwvgg6kdkpV}hK@MyX'
            b'hO<d9y>(VWegh*4ZbRyRnq)BeIdBMb?JQh{Feo6)$@vFl00Q1`G#uwSFn$K$vakbe2viJbK69EkUBfiSdCQUMZaLR!vR*Y|fxu*4'
            b'UCDd+3G*={3Iae!pv%~2utJ^DaC~$K{r)sQ`E+{boV<s%H*(H~A4ksnG5mka$r57%(y$%g1&cKdF(ZigQBK^lbZ+D1J_bS@#(5gF'
            b'=P53Rf(xf<l4VYBzss1y?C}5<zYe&A&I;Fb8v69TFK@jlo`s)Vz)j&@l&muuC!%#a9{qB7^689_p^tDO^DyCSkO`598D`BGJEL}t'
            b'Dy%9CRfX3iS;w;=eE`IK^|wD+Pj5pYzrw?xuGR)F>TwAVLM>PT9aj-B1h$+<pTk+ZvLw^wo)>He+yh9w2bMbE*GY;DM^b~}!FtJ~'
            b'ERUu%x$x;aMp1hoq%#t^ykMEvi0>BB3RcWnLMzXbh)xtU#?J1W7bs`)bvnspM7YSKMU+1X10d}1E=(U9B8-AL&=4dgNGWC%V&}O)'
            b'e7V2Rk|8Cb5ho<axLXz_@YNzoZX4^Bh>#=(p#r)Ek(af;1YX9^p)E3CjND6o{rchLXauwX(X?d30a=LKWeFq6qUjBOe1Ai)4DphX'
            b'5J7(bcsL%Pj7ilnbUzQG1<Wk82JN#^1UQQbsEIVeKU%aJ8sX<)n!^~7(Ajzdla#OVt3Iu2m^va{2Am}_sKftyQ#I8c_!-65KJbHw'
            b'BMO^M{LuNXBE4*Qd5~U*`2~^wvIEn96Rbk7-D_UV7eVf?l8lx^cIh|*4=1MY{Cs@!>oLLN(Wf2$*Avh!=ZmNTtWKwcQ$V7n(J+Nj'
            b'BuqMwNZgZ(%t-<24l>0&)ITVv#VxZSQL_Oe<#+1;K{$IF!t6pjfJlvDdXbZJp$va3i~)Ky1Vsnadz=6R#Mg}gmULc-HP)~sn4Bbq'
            b'{#%90Y>C*G@q^WpmX}83Uu#fa(H{XVVi;^1M$6TL*cOHh(jfpWVNFED#qcgk^MVeAP7*~VOnGo}d_F!oa=;YApQq!KllR1;?KV*U'
            b'IwawrCdj5tHXih+<I%grbFilrRASZ!*<7?n((qsFNLda*hZT|{aV8;{PH3~WqZuF`O+nksl2(v<-Tl_9{a4OgXMFH#uffZhi410y'
            b'$25x;Ao-{fOF5-gQ9QyEq`)*t(}>J3Ao=x;1H(?%)0=SCYQ#xitnnEt){mp%*{3m#XSciis@;9l?(O%Pk}wy@14vwj8QGk`O*a8B'
            b'VDDx3<zD!57QA@zYOgz)PIkln+1|@H`>%R0r!OXN_9wj;z1?ZJ`+7dx4}v%0ZqN;<-Q72Plb79B^WCefPUCRS_6q6I7RwLCq>;g)'
            b'AVdKd;AoMzL^5DCy;4aF!53%_vXCb+N*F7P4u_7sk+D=oVV*HR0r|kJfYnMCVnbO@sKbb@P?kAJJH)f4I2)xL5PgldEYCbc2Qf@N'
            b'0;AzXir`hVTo^<l^4IENC!N09^U<0C+1{A~zj;7lr~C5Fi~U(|57vDXuxz{=9h|%yjoT;3N52V3MX8$r%;;&dM1MdrJ=*kXx=WH1'
            b'xj9SLDd+<rT8svM-oQStu3k=GzYKS$ub~$-{}*eN$iOyv=udn^tb;@F4?c_zem*@pJU;hF|2Q8VpB<hY6QXgj%5H!#F#ZOx;(8gf'
            b'wFYyF&I3+S#H0%{d^?l1Kw>7kzA0u_w7jgk2CqS-SU?=yk;l*?nJUhW%s0|`boS}v@a^Hz;rVZlciKfHM#KQPo`LtLx2q(I^EPn>'
            b'U`|3jGiNf$!CZ$`xp-*Yr;##@ar6a8qCo@pnWKiytYFy@q$5SFDT}2+y%o;FN}EOCL^;4{*Ps(v^e_zpL8L<zuK^8c)1C*Q`sQmu'
            b'P7QYC{5oXGrTn)&`0Cy0Jy_)}XL$UM-W|ZwMhSOzKK^uY{%Jfs64trvIR#Z3IBa6gLJ`q0{8=>3J+Q&loGwF9EHj*z8Au!CBxFbB'
            b'Y69n^fJI{xWM~q>j60_NpYDO(8KT*#Isp&hMeohkRa4dvm_}e&zIJZIhfEM%fFf#CY&8}R=ud$OKs|+s5?K2wo9G{d-mejAqmNOR'
            b'nIE?Qj?g~ue#A9C{2XN;L5%VXGzn3ro+3?-z!g9P(0UC+=#(*Ow;SZ~h7k!6g%1hOG@1$5iUKZZ?qtyd1OON^GY|qO7@j}h7xC8U'
            b'y#hBs4IVOEJ38P3WV}dGD!<rQt>8KP8|WwZUf}o<?f1blS^&2<z`F(?JbV6}9qPSqxBD5IEwK?az7KIFlFtP`Q5O3AfPec-pdic{'
            b'%p#g(I}P<H{Ja9J@R4o>hINcz74mmZC_KxEw;06E)zwWze*mZzaFzcrVz8mZ%uk9M@dVilE;!lQ0crHY&mali2K-ULkoY?|IeOOs'
            b'Ym{U%%3*vHvq9qAr%8NmC_5%YexCq-U}HcF5j(&{X!u-nprK)EXJrQGj<P`EZA5b-h>KB9+>qRpQ62{AB7((Dt^v(uumUDm*7^d_'
            b'LKO=`h-YmSJ2|L);C^8HEGLnmC*P+P3FIc@jZjM_Xtf!O8<rw)e<*^__p?QkXTDD+OrE06EH9ZwBVM*2$0x0Zh<=}zJbuz*MpK~*'
            b'HUr5bxVoZQjlztbGvMvOnXhSFiv2;P#3c1+>@|cN*&pT@tZH2DL(m~tSL;}4H1lOH;#`kCQURHq7X%p(_sJS;ubbcw$-73Ka@Ru}'
            b'rZJ1=bN&FOh_Q}>>mZ830Sbt1$dzCL#L7IH!_Nnp>O74mYr-9Elpq7k;10DXhHXgGPQe~Pq=HF?PL$TZG6a!f#<1Bo%P_(QPnSUh'
            b')Z7p7%aX?kaXfne=?u&=!!A+l1Gs5;^y~1qGw0{g=#;J3;crJL!*^QivGydimq58R3Kr2H+^rbxTE#Ww3C9z+sVHr)=C~}GU_eta'
            b'qQI8eefi4yz5#Q3o!^ivAUG(ai2TmLkp?h=e;jh~@nRWFPtMv0r=NgX?xGBxkR6~7^noaVh#&o9cyP{^oZ+!vidG7dL?QHi%@zUs'
            b'ks(=jY&!yrvTJa4r9w}CL@VGX<f>=vLQK|{_u2q!3Jmv<jkyR%Ox_1Y$bwsbPB`LHik2OW)HWJ4|Gv*_!S}@%acC)!`;s*p>REbC'
            b'!6)^5nm??<O#K30yIDjN^^^aB2OV&A7wYNnSrV(i6a8s+W43|G1*cm7T2FvHK%AoWgZ>khwE7$91KwoF1Hu>^^sPp73P1S;piBlH'
            b'KOe>qdUWx6xq|5iGOf^FIXwaQ8Nc&@H~`o8o9I6$i@VTkb}%9Zn&F~%*=T$RlkqM71AviL$Xs-sx1bj!bO?Qe)igf*1=gBgzO)L_'
            b'Pel<nMOT`_Kd2?{!Z-qZ5WL85hHB%xKJzsjPEJm7b8_g#-LV2K%P38gRM|^eG(m-iFxW86lim)<&@g2N+=1T|e({xo;-?t-SAwbJ'
            b'q88>Gm=ORI7XT=wRoa6Ggjq&w2N(f`hXe8u6`Jf(ppOy7WR;+ryt;~F43%K$7ZpsFAV^-DNNEthGqUeH;O)F0emXk$$DfYR4?m84'
            b'@XpT_54k(PBnuFP+0O9vU62Q7D35klu+(>O!0XkE9ne0rXoeQ7pzk&{S!HrHlF+%u0Az;qSM4-O$P9{6ZL*febqXT*yFS7Lu`M3e'
            b')fKr%qp?36AAA64{@I5iVElR-?!B7N#KH2|OcpfZ0sE`H*RNW8yItpPv%^?|++KiyPJ;z)B+L?AXK1hdo=jLslRQPm#U|rYVtR`Q'
            b'BIvurcPOS;PJr;(|DA*gBwH<Liw9@`5)6DsXC0>@PRh*DQ9Ybd-0s1s>V3GFiR~o}m(#Ri-;BN3wxnZuY!hNBMQCzo_x8g5@YTyM'
            b'&_Rutc50@iykYhc=0r<)bWzF3WqSq<6^dm{aOFa`$g0S)9I&E*x*0D)VAQP&QgF8_z|z!uj=sd_ulIMIw-jG#IcPV8<hv4;*-S!^'
            b'<A3yUMT8l+f#D3@NFDZp!QfQzGD2_egD?w%0eXN5%sXKj2g#X;zy$Du4Vd58S<Y4rTNz~S0(ix1zjB0C%n1kBeU08KfD307yuDrr'
            b'X#f-_mUgkFV(~+ncAj-MqKMwOd1gzNLy;%p%vDYwt<HON;6pOQ-yP!C58IlCv_ydHXgEt8$|Q8E@vf_@AI-J(_Z>?kKg$#Z08OD~'
            b'5)c{AYD6|S{J}Oe8AJg|7`0Mx8hAs7H1&g$_PC(gS!p9ULAT9D3rr%|69I{@JD7diG@xi_;E-F4(NfS(h<pt1IqnR=qIoox@B{e5'
            b'=nn_$1xA4hy4vLOWVAkxwome_iM}NM(WjU&d%Y~#%*IWyf?R_-t`=Z+l-k<$WdWn&x6fGYemET-NMjX0Lq2a7Quh4h=h5+*lCOl;'
            b'7lbTo6=kmNfT0{NSGf;v<?haXlHTr!oS(S@t25ty<x8NyuyauvVWb~7dVDe$eYCr9Te7o$<2DMJZH8gm6Y#1?lu<~lr6e%8zbfeE'
            b'T~XBP?Bvt<fCTXs43-|f9iA!4?5fyTck*U`-rIXKneNYez1Oq7*<|)&KiJ)y%_qCz>)mj=*PC>M*WvzTe>&gW?@oj6>o9!fHpVBP'
            b'&PV=R07f<rOc>%1@*(p<X3dgx#|ovb9^A$+qvLlc;DY)Zoo!zmkwIT=9!!A~j{Vc&;TUJ^Rkzz1{W3iAKeQbB_g8rM;>}*mdD+`-'
            b'oE{C2u>)@5JeMYo<4+&`gQMZu*$830Ea(gZ4tWFD9Pad@1ZKIwk!)H`AYf;Hc{dl8TYvBMoA%!RUX#R4o+<5)LMPy$HI5K%atdBM'
            b'I(W@{bxJ7HY@6bo<R9&@B5t0R*G)iPBceciEaAUr2)?*CULpL|V8!PbM}=e-cq{Ba5FWHF(7hfFKbB<K;^7Bsf&fXVd^!5w!NA?&'
            b'yH79y2PYq4O`iL2e><lM*nicKG!6fJczC3r1I|7Sw0MaK`HYa&2nN<@oE-pT`Qst5<h#-F$;ZRv;rYoJ+2OVSa_^<T_h!Fw=ARlS'
            b'3W9+5yDvLkV88nB<&(3~PyWH^=!iP(QhQHv;d6)h43=wy0}KCky*olB;*D9L(T^s}J8+bS+BZahd)6V}>06rqXG>({n{WYcZT2lK'
            b'cTxqcSbTD*y;bnANP=0P{cDo_z>${d7iX`-->EN}otyCUEV>4x&THC`(S`EReK8b@2FjnnG?l<Mi-ZFFw3DpD*mKi~+eAMNP1{At'
            b'5NNL-y`7tNe9QKGn0kv~Ihh50Y1u&`;Pspze{^<tn_w}zZnOBhz@xKXp&9NGtfo0?d0!nQ2jYE-P#B{GIbS8^F=%7%md3=MWM_Fh'
            b'i&9VA%E5WM1|<=G1~1-EZs{i@f=Vk%ISYY4N8<R|dJZd@I_~h_AWB>WokzjKEq&S{H0`<OCP@Y~S}lTU=;_ui>ZdeNbmQyC$??6?'
            b'Kr7^FQ6Qi*1D9K)&~h-RBhClAXr8CJj}dhqy4rkhDm2I;pM&HqvlK=GFMzLFg2<)CwT~_~cl*eB^d)eCOLTx|OuamSS%rW6Ecyec'
            b'z-3OvPfC2;d=gwh`d*^~^f<EAui{XicCZ9;&maJQSQ&N8nJ?Dajm-Wx6*faLx);Givd+aVN5y&rONRF~L4LnePNnRNeM_kXo%<i4'
            b'%MmIAjtF(m=pWDhaW8ud|M~FYgR7+sDsP_+agYBo*IY3r^z2fL^)qrYz%0kXiXG1mM~;b#>(t3O6DKP|%S;o}00<?TV}wvTZ8|?X'
            b'1?nASqC&)^K?GilRy3~kmnY!W-e3qyG9wGZ%hFv2*U_})_;>KX0RA_D|0$2I<=lV?0n7s)0qXgd^5@V-D==Bd(CXI5Xn{giVNcMV'
            b'zYRbj!UrGSjR?=(b<9plIMZ$uef&uO{4ISFgp>I^3v(L-qy=ZE2u1sqX@L(OT^YG@%r5#2?DDChgnUdplN^>sS?_r+(!ga>z`F5V'
            b'MGdzg3_TaxxSC2$Lj@?rF0cbcMlg`k^wqGwm%cCcm+Hkp-dN+jaM1*GD**eKrfgEJg`s~3XG8JxW`YrO)1Q`{^QuEqWG!(1-h7GV'
            b'tAC{_S^ruQwEis*oL-Rxvz<XSX2FXkr*De8J;9<4T2QijmSG-%9uCZe4<KCGeb)h?BJPP{#3?9yP;%Kr2AT}*Ni_C9#nI=c!~?_6'
            b'03b$ZAkSf~AgJ!=H@yHi8oSsS_v@T)0+!>ey*)G<darPc{--dfeL6q`)&}j;Q3_|~{BtxuDYhgKV3uPt(gaYs#eEDi$EOR*a>D!#'
            b'iABt#6pSA>)AI*y;+yf(6r=*0od{Io+|M%Qzh{nTkG6}ZgE_h`4_f>L4F2daOEO28v|VxZn~~xmtBV6I7_s006Gh?BbC`l;SUfl{'
            b'c)&oHBl@5n{tNsX6?LTn(-VqW3y3%eTLcJ(dq|WSYv%M(^h&}*iZ(3{2@=it6^7J0+Qs<i;i-QB`U#wXv)|6nM;~#%t+$u}GCl?`'
            b'W_W&%3I%6+NHc9Udxsz}b_-3s-u@dPWRL#a<^L*eL!wkgODt{Ef=Pi{k123%*d>Mb2emN{5dcPlfRSK!&(6`iRuc7Ub=A^Nop4%p'
            b'$*3J&t%i(#{~I=?3B@!m&F32}Owc8Tkx3n8o*`a~ck6f&#kZz;FHGI}XS8~c7M5Xty7wrpCCUK(4|H3YF(w_k12P{)e^ES7U`Ph&'
            b'LwMr3D}YCFc69Rwwe#P!fOK?>0|V(MNR5Hrz9d_mjjUMerX;gA^0)p>#oBml6pUJg({fA&UvJYJ^QoAcLL>rEX`YgGZD^X>StGl+'
            b'b}nK{T^eYF@i*oRcm}2C(h4k*3z(!!+c&G&B1(c$5Jc96&U28pv}&Od2a@QKKr8daP?ZWH@(rQaZ)|9rK|#*wkVv4)bweXl0&pPV'
            b'RnpduWNRQ}#kdH*c|ZU?GG7o;ix@-5Fo*WeZU>)|UCG;|i)2b{<*8FZz8O!1r6&PMEiwoy#_nimD5c*W?4JFOLUPt}f#ht+9<Fr7'
            b'n_`dBlbUtWrIi8ja~=YA?&1G6^@s%*Sky31ZtL+sR%}2v;niBAjbdMdp;_n&W-SL}&9y;wQVayjV2g76nD~cs)l?)IbTcOlQj!H}'
            b'0*NRK2(lxDy(SN&%Q-1~xr#7@4mZvCM8ee-LH^#^l0S+0rC@Q$?ks7f<%F{8aQ4+c4WY44`7{HpPu#PaCaYrOjXI({)+z4ks?=w|'
            b'1yBT2Ai2ZQ%4%jUbz`u<o%s<v03w=W1XE?n2ni|ndp<ny|Ldn+--xeYU9oJ``EDf6(8xjSlR^PM$R@8rU&jcgcGR3%fPlF`-iGmA'
            b'1a@KPIs|96ruRZ!s_zO0*So-KXG1ly1f>~_&|frBa#}kEUyz_Rs1bWCqih-EQ;__R8hDN9)bAQ<IJgq4n+2UHV@#V{)-mrM0FABe'
            b'RbgA5tppfV;Xf7hH0e@pxQ#$%-+U`Rn1M%U0k!nL30zAWR67^RAj^&<gICCcmZJqk9$Zs^io`>{UM<24Zc&LV6=96G>u52nP5#AQ'
            b'jMLG$eRe+lY2;|{N<_uE-XWKLw^=g9eM{V_DpEPwA5l9Ua&A-3+$@6!O!^>b&{9r#EFL(9^ib@T_jw;&i=C@0Yo`bl1?7=rFK8Ts'
            b'ufrk<#56U;2D1nFg^lte$B$EDVA139LlwfOAz<z}U4v2(C?i)6NnOKoPXly=Bn6njfX6s>+6e&HyGB1ncgp;p5Pelnh^8Wk^TO0o'
            b'r!O)}rRN0BBJWD5$h4XA_J2WWF!W);g`CrGixuv6Ksv+Oapo>j;t@J~G8%lXoli1%uuV4+IBr}B_<^9Y<G9rsV>ED%*niMX&1uUL'
            b'ZYHZ|K$wulB|&K1`SRTV9lP+aAhS7#zla7`W6K)rmUPw>#w4agQ=C)HI<!&mm1Xt80oW(*bm}&>MA4sq$rN!77o|8*6F3JN#19U}'
            b'hz<^LTOK`dy>+h|xEk9fG7p%+g<6m%E&ar8pS*C*pIJT|OnW{1*-e015$2=$>EVI>V7_Y88V6lFNal0<Wt_C<L6*0d7&tH9ps%5c'
            b'-88>QO09%9qz%V{ik!X<$PP~i+~O<|gJVUQ3W%2TSo9O5*LQAT(aJyodCTX?hxf%s0X{bo)!19DYR4q~TMg}DJvF|08er)}SgSZR'
            b'W~bX4=gj_xoCHS5GqOdrUOJRC%_{yBLqVab)C%V30s1lMEetD@QqsW<gpHyL{wb0$wbBPJ<Lndi1Y%4O?X0w2%?Bd+NeA>qnC4!$'
            b'ReYsbjeGV%luA#O-2hm*R!2s=ztExib+aKoOG1I^nDAf9-Eh@0z;XtI+x!cTS;5%oo#6<5$$fEKHXVQfMfRgBVch?K+g9@0C^ggQ'
            b'$exiu=dc0Y`E_<t9o{9WKL)$izf+AYF{FrJ5XmN&xhObEs2o3uR~<Yyo(2zkg?t2GCw&0%x}sI!z3Kw1s9cz)vj!9HjhQW_mxSJk'
            b'891e!1|C`TN2vCQ?M+QRf!G2JVg+c6ac*X7sXVzy_1%;6R!3P7Cj3SN?Gd|vvaMnm|1t!Dy^>Af8OGBF@RWwxq9ZZ`7k6jqobo_r'
            b'Ye<@qP8ZVqz`<&@c<|98%{-}-U=X=Qwrb`Vv*7%j7}GGV&DLmOUzVNmfX*F|o;Jy#svIbD7*7snMy9G^U{e_q<sAo@QrwAGf8ePY'
            b'3a;aZFNMdca!CX})l^(tevic!LTVuvu~%XbN(ZC}A4SmNuNd0P<^mDYs+R(Dff3}HZEEtU21q@khLE#7M|$}el&x9qOgX9P^5`;&'
            b'd?FDj(ThtkvQ!owmmPqoUEbfDERqSMf$PXCx)Iw*$`lNouB1=^%sw4Gx&ZuefnQwejY2$?1mYQCc5ni+0!JeD06ioRS{1%6QBUZQ'
            b'F(vTx3uts%PYfXleb`}g!>aO19Qy@VoOp!$euS=`FcU`R?$Lm3df~dyW$+m+=)Z#;C^F>!Y<I;OA7*ke!pWiPb~=w{;$7S6nK-EF'
            b'_?{{0F-RCb&HV)u`4R-rdg(QbkBoOHq3{!HTCxr-YmloR{;=eS;;2G{l~^kyi-<nVObf2UZSq&khQx6S%g!?x@PEYAlOi=IV+TC;'
            b'bF5QyAV<q23u2vO_^HJw*!0OwQK%CtJ#n4jnY?d&+T-ndiRK)o9h;{FnX<yXbVSlVC+2NFvD-qpe3G_Rb6gUSYr>DFanwvDq{t~V'
            b'^I!%LXOZ0V9x;h2brOuvJ7)K&zPWi|>`ao}rA=LK(`gt!=o$R|k@9fm_;ROo)Uk87%Q??SeEN)Jd+8J*9*D)$X1F1-qBsN}He@|O'
            b'$*a7VfoI)2#hH_w!YMq@(`2#0<*Bp~pGCjJjD43+OQPrtKTpF|F7Gj!#YM)>DLp(>=OJL_haed<+I%sN2?Dk8WH$^AXx?cYjo$q<'
            b'8v8$we#87tXz>RuDDasdT|N|ujx!y;BM%yLrm@%-{#b!^q~ECH<?fTt+#wdA?HMnb{BC8Ckyr<|FO)JL6)a9U64J)GQlO6%JdpZA'
            b'7_;H?w;#tRqLi<$?t(N!jstol4J30oj4At-JrX`dP#oW-$?KbE#wm;;d#|tx(cL!=u;Lc-GgT+(S?d`eC5AunxP*^%z`rAU#3zdJ'
            b'4-|&aSkOGfHa&a@|L(%S-~>EtlG%W^1f|56kCO+J1WyOiu_WtcEGzq=IdnQ4tB<te;*wAm4(kWvKJzqgU2`3nFHiu^K8g9Dv9pdb'
            b'7LtXbNk^V2+xVc_?-$c;Yw_uVkNtw`MGk3pe1DlHzHeuK5+^`2Xx&lc=BSPy1u!7!A#_%}l0KHbN9E5kK2{5jrczG$quD6;4jm$M'
            b'BA&qQ3J>V`@yY1&%iV40O63Z%nxjT)9r+@FeX<ul`hRON6_&18!9^0BI&90Qiq+zx!Zue|*c6X3&kf<q!p_Kmi?WCcwoRqymeF0a'
            b'Vi{7C&7XRcBv}vy{KIcn0cb1r0$cERP18Q(54Us?VXZW1h<QZUm<A?i;PaZs2y7b;X!=x9fTyqawxKWw3zw0gzfiSp>xCNC^h3{8'
            b'Px2Mcnci@)G*nSi%3_XUZ=smCveaF!*$pBTN>v_^c$6|Jr|eP7Y5t&&`}KWb%HLCE8JS8!cT%%}2B3;nm#)2nR522)Lxr*nPuqkG'
            b'ET@J6d#>?d?7E4KH7KtmwV;i1Pn>>8exzlMYap^nY^O*Q;$rS%4ap9rBs@_}LVc;}Al2;{OC(4bgMEX6$vT7z5*PxmSuY+<$};$p'
            b'DEu-h__Gak_-b01^Z<yRM;dd#yE}W@NE#*na@urcv8T2(xYAjby<2Dsl{eJXtHQPboS|KZ%64RER(U*2Zgoi-l!s3)F!@)tbtI>='
            b'1{#jh1m?HYEHfwb2bjj}xgD6u-R8!AJ8Hg3@(>i12m}YX9R)*Qih*9>jJ2ikjpUnFlX|I8qp_N`K}y~sr}Y}atuKm{*48Comk`JG'
            b'88EdF;}|X8)ju7d{4&yY3$X#1M>C!0K<CZTz-oiSTFF^YskcT(a)^lp@dG&_Y`Hj+u%-d3f(u5H0Spr@NlKfTuVb#m$k}Y@Lxk~6'
            b'XEEs4stN~{BgDxfXkh6n=cPR(I}<?so=i%S7Y<akM6Rke+5yLQY30vh#hg^L%SOg0t~6Y9-jRhbv~5zrk>zXTlc9NFML$^Km~zfg'
            b'WxDEUt+wX`WbU-G_8E&>8>{ANfscySk@}PVJM^om>l>=a^sCvy%oDF^LoG%WE;fv5iy2~$%pxb5BZQ(=Y_WUU{F)BG$|%vzOJ>PN'
            b'oKQ&oWt6rXDI5D_W95szX5kkCRpRfe<x0V-J|EjJoIqHpkK`Q)+D(yIvb&Netj;!ytzu6D){**vl11<V*R(9~lrRMDC9@e`G;-QD'
            b'2;A2;8Wo`12IS_J<Tfpl6#;aaR;p>Eq%<BTa2^G*i~gHS6{GuV*&@k1%G&fPX<H*K(a(xh_)=7}ivKKPenvq~*wKt?^|HB;k|$f^'
            b'6-|TXjErCMIcAHqFWJ}-Nx%|B??&eoYwK^Qo#fQkP9j@Z{2=h<3w0D_y7-@w_(3J6bm-37tqoK)Eu(zakUFtPt0ECR<AGIlj0-Vn'
            b'$DB+KLY6~>@dIb0;E?33C5F1P`teA0Evu|ki%zMd1wo}^TSaXwwZJelmC?Yn4Om(m@KTw06qHiIK~k!!W*(Jwd!y>X-Srqft}OQh'
            b'jZMSYeU@=8O#V_*Y=rO!!GiKAGOi>r4Ht_YDigPJb~XYy3p@md8!S{*JmrGDZQ6v!q2rtEmNutt#SDYVD?tf~_i$neeCnYr@=5mD'
            b'JXwCmr>zS^)V7E>ndk-d4H1hi9+2q;SuaR@7WP2VN4-(Nb+j?yXM9u)aq~yy2k_ibC*+yvCk(Rrl{nk_X~8Fc3ZuiJMWWv`>@8*q'
            b'DK0jos-_9`x23b%4|o-A!(iiQ*n#XE4ODEEe{cMR?2MAMW;}(g)!ytURozSWZ~D-$?New)-7@<qM5`T1*m*?%&@mWrN>*uby$t$T'
            b'gN!nMK?R1NlRTOhVs~4|?}J5FVF;GMvW(~MBQLQh@g9{x(<-3?9~Zcau~j}7`#~*p!@QnSd|A1C4Yi_t9s3xN(D-^!T{?mVwHX9;'
            b'x%q23glgeI77RYMKpP9e9KVL$=aXNC0U>Ux&0ls<LWBGxTG>dn?18nUERK=nWgukARMivwTZs?YSp_(ZZ=8Qs6omiQDOE<Xd8X>~'
            b'mX#&cvT|cumE0vsCzj3eG|%`?b`q`R2s3lj{7Pkm@-d$xLs{T(a&oJ<T_NkEhtZPld1W~Gv<+pvVmqNvBBa&Wc37VR0%lErGw%X7'
            b'Y_6S}ia}B=s3;3xGRyw*r5<{dO>OmHH>kx_)&-UE!3%9-vs@8X&{x&$b&<=2jK7dG^ax<Ptt!rQwRzkELf=j<n~)18T2g&R3eZ~x'
            b'eRYv$sV-1l&|p=e$C8qnm5RqyXZqQuk1DU20EJ|x!CF*VXm&+p)z~g6U5-O>qc+IoHsuOkR>9&SWK7k<jaW?jXRzK^f})rL=$mEq'
            b'N9gk<DZX4xk_fvtmGyQPQ5rH0ufsn*ce+};MWVEi*chdHIz~-WKf1|!9<44$8Z$(STrhpwC|nr%W|YrDP(2O5CAg?l{@HD2enAZ3'
            b'(__zU$fGlnQv;?=I6KSG6YkPbRbq<$;MZo!<rvr#>Xp8Co>u{I;y59VT%6h4csLH!8Z;imora&jiaX&Cn2<b}-gwOp%n1BH)#WtK'
            b'3ECG|aIgxW#Yy}}m?mBU(2P73-x~-j<>xUcP(mShXLr)<<cXSrRtW)>#%YK?N;fdF_C=;s-hQWOhgo2*dB+BOx*NY##k3|-toK#Y'
            b'ocGpsLX_Tm$0wbxI-#Ij0kM!uS&>>-zmcRbw84vK`qj-tOgp4aCc5cE6{lKIL1J0JC~B04?QmUVap<s|={j2AY453=+{aBBgEZPC'
            b'`U`wWen&|Wk-ckR9L<&EN@wBhxhoDVaaC}0cIpl@m2%cf&{~+(k=*-u)z2avkR@AiBWF)=jT5h&bf$7EE}P{b8V*3cwNu=9r0$)h'
            b'Evv{DcLN4tC7oq!45wl<&s|ErIXwQ!37kg;`?vJT3>jeZXNd++9nK1HYv4UrFd}%8kj7zl%8IdeC!_#%6`*Tig24p2FoDX(OVRjB'
            b't#_qXv!o6nF$%^|)DcA;Y>_hW=6z&3qJsC!U}{kfKq>F0a!fGwL|yPk#jXRyvaO1-iCLj)M0e^SzN355%CQ?RcipBaQG~@vv92_;'
            b'eaGDp4Yy?qhK`vNIfLgCdCxG{3t7VfUv68Z|JdzIa>!=&B2B1nb6aBg1t?QI)q+!rM%Ap%GOXG?t<qkyHnkqX9Oi`3OOCFVRv+&C'
            b'DbLf$nR;r}Y&^&*y@5XSbPSf-T{xb;+8gmZPal1;g^+o6$|Q<ZqOy>GtYg67n$jFC1>;utT`-v}&Y`6%A;-ElJZpWEb!u9Pu<DB_'
            b'YSm%|RKPSiUj?3_IXbWfeKB;7PYlwQHg?LCwgVl1g65*v)kO{iDmLB$Hl6d$6425pV8IrIwqEfdfu$ohnx-B<u=9E??%ueCu5Xu9'
            b'rmSJWSRi0+(Ef<dN-6Y7Wg2?rWP4js7?#kMeVhtrY;$-O;7zqUxfwcl$|xSFZmb|F%MVywDQ<*#CeMjGS`L(A*g)ff=f{EOav|DR'
            b'bZ&_Oy_ISV%3nmcp(b~;l7sT+uQDu$xu|tDdSzZYh<ViV6C`diM+5R|>|u%!ESuHZm?^F~EZmFj_Z^zX$yWp!jUG_KgL3!i+)B&Y'
            b'+nix<iQ&#9ypCcfqo=?bjG`0yWpmtC5Z0Y|Rqe28Fg$4`)7oTBax7YokiPH`Djq{Ap{MXi>VnjA&nH@lf9~ebD&5iTl#6Kx8t?`$'
            b'%iuB$)9^Xq0ze07i02N1&rvq$H7f$9YUhOk?Er0xx2st!-%D753<ke>AmRH0!!Ja<4Gt=g2BDBQPj+C3PuyGx%9k}XDMBpfDlwk>'
            b'1gU<g8VSP($9mWP(aBGT=l<z<ba03Voxfr{%3&;426GeJFa|!Yf=m^5tfA73H@LUl@z|*1qA#5^m);lWv&t#OI{O~eT)=gj5T6qY'
            b'E!9uWzC-Ozm3D_Ou<$EG+Q=O+++;$<6+O8aZDW@z(Zux5%FF+<PCYA#raCRjx>-Slf+V(xQzo;>GG}8;ek_;Z4iwt|9aT}EFktp$'
            b'5~XGFM2(oMqz<s2f24n&`ZBjO2?9%QfN@_Y19ms*3#?bb+F}!i%}w@;(5g|{*(^sG(A#40AWnKs=YLh33_x-4W|eB@{HTKHUtZhw'
            b'X)EOoE%;(f7DZ(nt1b`47$xn2*GjQ31quoW-wq9!6N?mhlGK5tle1AJ1<~CCc59MG-+4sTpE}M6YX;4%BTAd0x)Z*?4Uc?M4G+_u'
            b'rVfjPRw@a@xaY7QaFwuD%?iu2(~e$8mHzT-CI(g=W}f^}PxY^fJF3ck6oYKl@4S+qHYgI%D{-33X8o5mx5kHhuB++stqmGdgr5OK'
            b'KdTE5)PvFTaiIU&&`_59z?2U4O`kvKdHM?FY*gO+56x3KQ27-|)W-xi#|Hj<jG!7TD6rhtT!jrZcWi0PlaKKZKTG_1kW%Rnr6G5m'
            b'H=S;4>rhl+ffBYPs=#7z+kh6Z1VM;qA>JF0=Syos0GP|iA%c2HPc@SyyR%gG9OSiH@|a?k-4ul_C@#*e%v5p}x!{`>J`R66JlI@W'
            b'%cX))sxMLyNHVI|tg1(FsQ`@iDNG~;(xDPQT0lWN-9nrV5l-sbq+%?!5U`wY1%f$RLIqQyaO4d03nNnj9bMZNFO=ues@*&+J~xod'
            b'>N;F>$q#Z*UzmoS%A9=XY`CScZ{8kWy>L)28^lyHta2r=j4i{N+yR^=Tn3BI`-f>3Wz`BUJPxYdrLG>7Vmi=yWpjz{MPt#SH-;?z'
            b'Vtm&kVo1sgp<Dqsq2ta<TngY=RsRsh;~5{f3NfowZn@aLar`>C#>Ep{8|SUr>{09AkhVtn*jVJ1_aqFWSr{y~$pVSBTSvMLf|zH-'
            b'OC#qY+BJWQQJWwBB{A`&&#x%|2L32juMj7pZza05|IV>?UKdgv0>hd2rmJJ`<waDza3HNONU2izj302W8HVb0pH^vdj_tQ;f9rY|'
            b'!Gr4`E!RtrA-9}f*nZLNHuX{eZ+kNe)o}afnK-^g>pXBRvQk;TB6DRJt8;R+grsOAe$t^3HeHE@uUy^eYwNYCVy1&qF;+8ir=@Ge'
            b'RSFB*B?X&y7SV<hHo7mV=~N8Hy5>tNuSo%3bIjXs1Jg-KX4s_mY9>$qZCaJ|DF|j<^iWr_Sbf+OA$%@Zhfxkt<>IA(|J%B9>56Yi'
            b'qClHmT}@U$^-2fv87QBGC4+eMjzqhi*RAfy(Dy$L>WkuI#ccR>QGd%){}MNBB{F4TxltuHY_o@J(NO<VvwIe0!8Kigsq5F!SZCJN'
            b'RV4>$?s|`CFVO{TAjj&F#j-zlnL{nBC^zAC(=Bby6eKCn3myF<%Jl1dyDZ<ZVDv>FS5erYn`mwJX<-^NU6u~Ku+rg5<>1sJG?WOg'
            b'MtDR2n($<R2++#plefGC>8+XBXT8M*YP7(gMk^XHU1O74+EUR7ci!YrTLY;}?5N5ot>P~v=Z%WYyidoYcZcVA3n|8Bevor;&N`-w'
            b'61lPT?5oI%Dpv6wo@KK|Egl>pZ2U(ODr_OBGc7ubWfs~hBoIH4<u93T>OwU%Pwy@Yl3bP<M`tQ7a&5~RxtOL19S|rzua_;O1y7J_'
            b'A#+>P0;kY!MRRPlq}k*qJi*30j(H8r-l{Mqf24TT?3@`|`V~Gm-s4K(m$Wpj4zCBB=B8-HPX@-@mQgRrj)-xq@>7#gC`h>#|J^dG'
            b'&|!)N$MX3mN}9I|W%6BK@NG|&JY$p^t)vs;_M4=strrW_eN?DkqmN-b$(B-1LUE3C9jlNKl}zSaW%=$5K8-@&35Kb#4PEFY$G32g'
            b'8?WDBkkob&t(y99DPNAOd&&UafR@rta_@i(OpQ**qxXmZa9F5!botN(hp>!d52dk?so7$$*IE)yo$Uqc!RH~cD_E57$303f;#BZr'
            b'=asvmL?r3pA|icdcK{D5;!cgt3WEBE)j>gal{Z6^T86~-ZkVEA!m0s{A|%5zZwfv6(p=rXy6CHh3SauPd<z)!(q<qs2&+py4b^7#'
            b';A=j>7rrJPe4zg}H33g1dq^~;;PJYBYm!YzOsJ2pj{IpVA1e7PT8*Rv|1iXYi<^KL_qkXG)04CI!RaSQ-LncNAYD0+j&d0CJu{>W'
            b'%u8P`&62a!+@~Ar0VZY3DSc(QwsZkqovE#=+qvY;p7s1%an-@^-Iih}9fPnqk+OIceZSJ?IygB#AD<jK?}vv+BZ(TMDw(}xWl3H~'
            b'^}=ENsBD6%oNaYA=%dL}rtMW?N_h2yy6IC)zcOe#hG{Yf+G-u=IN^!no494C1<%W?;^Ls-({+}EmdJ0)seLZ`0dj}YiVFjktv^)@'
            b'hNr+XMq~%?>unLRxlKbvGT(IV+3{$2q}`QbP}&Ol&bnJwvgA+4CnxXO{I|y)1IoFPNsk)^auqdSL#|TszoXjY{2iBgL|ua!$CyQe'
            b'rY=?45qWQPFXf}@LUFujcYKcdmgrasT+7+W<<KWZ9pwJ&MVRv~Nvxra!t6Q~kP4?Q_%<s6i?>Y0t2DvO%hhm^$FM9QPBmTNjD*{4'
            b'yRi+X{LMdOt4VP`EV$>(+zo09e6|d4lb(|bDt;E3zRmv^tvRN`|Bx}KChPwT<Lxim`dZtwlV1IVeRWwM$0zOg!=FcexywyNW0di&'
            b'Wh69kdu%kO%h~81Rn58&%+jzd;RU2bq2&;hx=igCywBu;g-`kFdM(F7x~4(EN?BgRnxq|P3Vw4PSrJjTFk42b#Nc}XZV44Rc}=&~'
            b'3s7}uJz+&7#@k!2W4gQmla~Q7mJ38J#+S?+TLUW4TrA$wt@GAK0$M~=tlzkE)qtb*H7|QAMC==%!bg1JodNA{gDtJUPP@<Dc3Zw{'
            b't66w7@zP~mi^uTeiM4`TfyJn3kb=0rJK%9MzE5Rks$R>zSA+W7yecGA;kvD#ePwh{Pf7_FAiUkpe~j2-KvB&?<nvn2?pFv;`(KP3'
            b'n^88x6soezB$OdMR;cl`V~|3?+M{l=Rdt5zl~W5!7L&3H`OcP_g<D+TXw)l}@|9fiRIHi}ys;^YS{i<N8Y|bB8a^IA*}Orgb_Gee'
            b'S_CDtSS|v<w?E)`BQcVABXF_+DrK$;;}YFC=yW{#RorU6ss+RzcsMgDIVm^S(=GRMXi`_$=k&f>q8OL7nuFUIe{{ZA8mv)!ZPQk_'
            b'M7I{Ngf`iX2)S{)q<If)8E15?_2o~E;^?AUzkEs7z!JfA#o!Bc)ED|mU322HYEyaAuEC*Hx|?>I3p4McT%P$-pS@?-t)e=<#@&;M'
            b'jktO>W<KxYYKcLtrwas22q&3aW;)#(gI11!;|+TlG+*J$iA;3F_a0Y*Sh1gusgT*RvjTq?v}O`JNeb+%E(rqv3-6#7m^bXZpvfLB'
            b'B0jns%whWOgLH<r!LI@5CQ&4aLPU77H`j%i_5DK=SRMlOUXB`(Z4X8^X>Tdr%#ZOh7O>KjTZtkm<$4dj`GBD&tn6}X580fgy2Eig'
            b'0p(Grlt@Mi)&tf#BlkQiv#Z+V5%UIjv+E{RWKbKhTTy>1vSenU?m*6j>U764fgvSvG^Hzvwn8aPY%a%J{`|)R_}7w(%yNX2Ey8f+'
            b'z3g;L3n96+N@1wSxH(;O?R@E^roPZ<I}&Pn3>><l9v!#LqBI-ibaJ8DEPuu*Qkf8W!ZmnHfvcJiaS&EmZ~JW9%_?!R0-aLJO32MZ'
            b'v-Gyy&Gy)!1!-GT1ME}dPlPp}KR2Xixw^d-<)nMPz8LDP`A$po%|T1+D?=BQzL0}jn|rr3ghIHgJ2rk{qFPcyHQ(5X8o$iIsqK+K'
            b'yGN4&E%Uj>OqYED=lXm!#W9wXU?jNc(G-{zSty0dH$LG3o%t!#!&K4FoiqP*H1>z%gAa$ljL3d5zLYrZ%-8uZqK27u0hnh)d(-KH'
            b'vXw@1*qSxFj=Y%Q@!->9BY3=%1iIMkyzafRwmdm&%EwYczJh_cSd`t4*$#uP#jr4ut;8bMJAF$iRj6!4JvT_2|GOjNDxb43JvT(d'
            b'O`Vn8_W$}hQ?2l~957W@Qe7iJB&<xnp(8lCQ4LfFVVzgqt|aIwt)@2O*brjtr#@@jRgPpSKA(uqgbvydQZQDcKy;AXV2*0$=k}wj'
            b'wGnaJZt*!I#iM9A%p0Es8>ZK5<d#!<*;nZ`8F}&>TH#IZ(=?*OTm#>qB~#z$vcqVqWL{IB*k(wbV1@w!2I0AQwI<GKOUI}se`$J?'
            b';JrlyPp_hup7!P!00O600p5g*)xcGk2kR?0X&aIzAV`<)R>0z5w$VSwgDv4a0ChS*QT}f9e)#F=+#i2B#_Qhw@yW^g7A%eH4a+eJ'
            b'gI!ezwglvHs0ar5_8hSzKmP~y4OI^Q(M0&~W<bF;1E*g?ur{jnm1ri!7r^DKDDKx}b108q4F{JK-!E<UvUOJ2*V`~3l+t{*onI)*'
            b'Z+uVV5d7oSALz3^;srn;FYuzQ;-0Lh?d*a{ss&WJS!nwwlEd1fnDl_E_iC)z{oML_5jt;E=)4xt?#t~=PJwk7?>nI-?b32W9TmD?'
            b'$1mvMv|&}XW#P7!8)fVjAXZ!nxWYzVGu@VFK^$t56lmluWFwGDcj5XCj2$U0nyH@GDz|t@Bq9$%1mY5eJnFbRvo@#Y>X4Mg+gc<|'
            b'@j6pC(`q@|;Zykdj$qjlWy$;a(dl6_y7}nuDifp}^*VdV@^Cr&9Od3zbB4YwFN&QW-lQcl3D@qfWOr6CRI6>pC1Ye7{L<TvnVZex'
            b'8f5e~44cQw11p4g-`bOXx)AFE9$l7P7kbE>wv7#*fAO*D`(tf%;Na)a3lrkD@7!^H4KPcPeA11oT&&C!^I8qQz>oWWda@X(3%mng'
            b'_*7LD@8%ic<t)fazVFu7r)<<tE%6@E@c#f1B*{V'
        ),
    ),
    'rc64_backend_encoder.c': (
        '5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6',
        (
            b'c-qYxYf~FJ^1FY9s8q3jV8FnVhaV}9O)jZo_i_nWb#<w%YwQtt_Qo^584bj{%YWZ)y`|CE&cQwW0b{Av@0Pk-dX{AKw2CPil`M|w'
            b'&EaCwdS;I$8LN$@$;=-slI+$UTSjbwBdw!ui*(5+R?uh(P~`lyKT8;iii9nelqGXA&tqDUo4g=wLCH<>mB#S*j`qv-a+as$-G_@y'
            b'5~Zs}G^6Ygupz>009{qH6xOebd=|};G+}GNm9Zk4Gm?}f&j2JZ6Nb2M3i_{#X7e>E@_VuZf&%*iBD)7(2)(0)m@TU%XO-pT<mc1D'
            b'At4uvZO{iT?xK{WG`gcDp;=ztF35sLagpasM5Y9JdB#^1QFcp@E`NB7e8v^9cE6w*`Sa!F(PavA-<71w%H%c!si#xGKR9`T+*c)y'
            b'50#{VsN@<Xi>M@7PF88efY7pECckoZ(KIeel*PCds23DwDX<qOQP$6B1zJfF-BGLc3=ExsZa!X~pG_~WKD@j5)0=DZedSJE|K<AP'
            b'inQAmumnF@XHb#Rh@1{OKHLX@^1;qeJETtrl?VLx{l`z&)4yJStRp`65x={*dK&;c@#(*N|M3Tb(j(C&r$E2O)+-7N2q+M>|67Z|'
            b'M`0ytem`_a7Rl|x8(N_rvs*Lz0tUNr#3C_-Mt~9gat7wnDw=~^x(ioP!ICHq4o=b1Vg@D%EsDG_KSAoGT+n#fdLYFdSObT~;2A!N'
            b'l!!_UK@^qjTLcBJry|tytj4|MHKvN!l0_Kz2*PIyT0&6Kwk;Cr%6|?N4f+*)xFaQ#V;Iy)feP8G^=?5SXy$ZDGbUNQA#F*dKjFM~'
            b'03oZ2j2v4KNR~{<r}zJOGyU=6@8H&Y5fW4!;m5^B#$+(mb1vqGmKYQ41EhnK^n|pA3k^B4K$W1HZAngzlK?Dv9T<B>&W>N6S7P=3'
            b'B1tJ}>&cO3(-sw((jb<0j>t)6!+HF;22-CIY;FHELl3Rm%I+BOx~kLG1hj2!<qn`q(|m3$i{5*nF?Ebi_beQbv7$qX%jh{;UEyQT'
            b'Nsav;ba>q5Ni5=HRWaNbn&Lwr9{{7|X!dQ7CAsfdI_$LDwOQDBVJ|1YlQte{`}m_1M#ML;UM~<~HUJrq$&1SOr#)vk(DETMhxwY)'
            b'KReOW>wU33-`Q>(WIwK&rzOK0=3$eKrkF}h<$3Wf26GIpArZidO2eEWLH75_o^NQQ)kdu3XH;uQXo0uiw}j<Vc5Pv%a|km=+c@J$'
            b';0ud><xX_wB0dL|*xB<N2cTmIs2Wr8@zdX~t~|HHa}0>6RKy7<eL5-`91&s)XJx)_3+eJz1o`2&$|6=Zb1Y^c_oVKA)TvK&IGg%R'
            b'R9uShHV3ykJ+KaHSaf=Vo1aK&%LDHa7n3RjPN)P>$7Nn+to0blC88vY>DRh0$nUUi>`B(pUpc{!7`B))Q`+ABLDNZ%tM!OVi_E$c'
            b'sb2tWbGAhM3Kro+<8+3k*W&|r#Q0jndSFa!HhU6F*G-^JEUC4!d>PAs%Fle=7XxTAuWDIALJxJyH0AW{{U`)wvNs2U<*c@y>B;*f'
            b'W}!tu5&SH_xhW|NOjoN_Xj*My38TxQvx`fyN8}idB!5lDXb7&Cj-xhXvr~gc*bOi%v*EuZPv2-g826J6aggNqpK<bwL4W3^970vS'
            b'G0~vI8X8Hqt%U6eVLHCO!dM<0tq6L1W2?9_ynSz0c}B)e!6U|L`8-*4xt;9;kmIqkR)J+;`H1ZTDih!}7Zi5WX(ftM_$g3#J(fCT'
            b'GJ(3#K2B_>`9_4uZ2%+c(7_6cu+6qn=6T>$ldAMo;Na^!8gU`V4MD>FbT|yTpi6pWN?IS-Kc?LbLZ8{kwlcVT5t^3NrG?B+#*d~o'
            b'@M+aF(zl3g02Q9s2y2P3nVgcz)b{)PTd0n_?tlXx(u<z~sT_8z4m4A{DxzV5)JB~%uR3z~mKs|l2n^ekLvi4Hr^@dAbsH+={&Rn)'
            b'dkCan>a>0h@K9wQhjScpe*Z@jF(*&(BpB4e?Ofuw)-JBm98Mt%9Bi<g@8Yi^=0RtJO0V&@!(gG}yLO<IvrD}fm3=Zeh=ze<bz6O='
            b'Gp{Eo({?D;b}84~aH-kZl=oZw_eA@z*Bx_<3J|j`07C98W*VJ3H0`*y%N=ISb%E*?q22T0J&ISdlEOk(HqvY%aP0qD2;L)m#Z7E#'
            b'ah;M%FO6c;G?BeW(v|;sZN%A{(W$BvBuIZ^H#K=eB0b73-}Ne!khQ}I_Q`W~tl9`HPsI?kI2jpE{63>tXOT2Cd7K$6Uy$+kbBuD$'
            b'Ekx_SIpc-f2YFWeoHQ6r4ZU`NT&CIqfxhv9-7|J}b>Aj+-B|lGcU#!Y_XL_+KR_W<Cxb@b)NW9YV7D8zb2OpXEk4r`cWi{Uo^Fln'
            b'nP7{8r;ghG+@g4EQejzRwS^o+_=OU>1N!uK{ay(Z*xa<joxX^&<)*#}Z&=77JNEREZjj*j6iStLv&F-G3RrW{vg_z>UNJafb&I00'
            b'+;vM`JCtP!q<_05kX~1Uv#8$m_Ts#$Fb{#$>xT^mwvGW&+gT2yTMFniy3nrG#i~yA+pADG%5t%m1=VU|r>=a^LE-^B1uX<<gVT;&'
            b')i=UTT$8m#vhL*;!b)ygsquLQrWY##7wD))j;zV~ttB~f%juStem%|*zORI|tve|~&{4fS_DX`9iw0Xo#)yjhG1=UUdD6bX)Aj^*'
            b'?+2WPw`0!{8hQaPRHzRS=m8*aF1Zc$|4|q3^CjV~@jI(Hc9_^y-+Rv9Sa>@KcL3W^$2ZmRL7p4y_n@}wwfk=JHq_Tap|`<qE8K%c'
            b'd*gMc!-8L^`wr^6P^d_My3{}(vc&4s+NY1X_p}Rm_q5-9NwMQagm?7I)_H4V<`~HOwp>qu0GPQ{Wj*_QqU#E62(Rhz@83)ahVg!R'
            b'Q=;X0hTCx1;TpqTpwu9Nn>rDKH7%s%v`<a~$)|83@ha?X1+jx~rR38ntdnliJObnQS#G`A0vEcYMTztLtrnLK-<%Qh2A^#qvLr+Q'
            b'<;(FA6rfz@431CwHJY<@jjy@lyqcwyKa+y9g(a&r!AfIIBC^7b9kyY@?sNVoY#Fin0&iG<s+J$t-<1OM`5AtL@m3MAfxJN6AS^AC'
            b'C{2Fl&&KF-l^0O~Xp16Rm*Uw~Q;KX}uYpxIhmer1>#X=+MbNGh<1qVQj~PdA;wsN&XzOL3-nOjR`Ru@kF7&kLPRnZPHKU{xTCWxr'
            b'(BX05g6%%ZGHxK}QZG~?NV9VWW>Fetb6TH>;$Nzg*$;61&G7L#t|jGNlvd(k;=en=TsN$`K>sl*u|b6I4iGK(FhO9%wHLfg#E-9Q'
            b'NaXmsIthzEC`Ry?&_5JR+7Q0Z)mp*IgbWUk;d9b>^=P&=CkqzKyININQBtR6aEK$`#_4zQZO>J*Ny_}3Z_v2w!@Y6HGVUFxujUaL'
            b'<t2PxprQQ9fS#;tQ0c!i9pMVH>dd|wb+|O^m7CiLY&}S;Lj2bw1tihS(Xo(wJunG!6uf;5(X;hl55>oa9S@U`!9j-5@Wj244qO-3'
            b'<#G-GuzP}{auS=913$;uF|{~vOfBq=wqnBWYVOwA4xNyp=}>1|O}=jECF{<$-t;(5-cgxqhjt%V3CzY;xc(2Uvpsb'
        ),
    ),
    'residual_dx2.py': (
        'aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530',
        (
            b'c-rkfYjfi^lHc_!5N+))wIeB(?06=+D0Lc-oN=vRCp(ku`cw=hK{nSEsU@kgoqhcF>jwY=6lu#z>ONfERLU_)pwVbF8vR1M*=$}0'
            b'X(mMwFT_I5lZ6!dofOOPQ7*)B|KLgl>HIGI46l-O735-*!H3xsd^_wmo6SaJnI<dY`^!zfNu}?LaJ5d-900~i9^_#XXN`t>cNb)L'
            b'Q8?3&(a)<u|DI&}w=7RL^V~dbX6rPW%glVw!<E#({~WHDVI&!cMUV&cD9AFIX(Y@mZm|w>9AAIEghzv0yjiWEM39Mi-Du?L)1V>X'
            b'pZYjUlROF=@^LQLxj3Ph<1|gufe^nFr^!5s065UGGYjVTVii1zOh$5^6EW9e90SpgPESM_XSoa(T>{Qf#Za6lu~dZY&cfWF|31tw'
            b'0bUs2(qL^EV}36eek9}D{H`rFF~0D#@D|{!{_`MBLz()i#BUlBwOgjaO78iLS(5sysqUDp)-uj$$^7KAOe2_u>h2M~XQS^<j*vu0'
            b'7iX8_^Q+PI$;G)(Z`;%$TSsAT!M7l`8!a`CjPo>EKf!o%xeVtaJo`b~^O5)F?tn+5%bpGYsOU0Edq>!T2}56*ePLGX-u7te{O~{n'
            b'KL?CH%Ma8%Ew_^~yaf6^$nf?qufl4gn*2a5YOq-L`u;-@XZ|c%JUL(Rt%YD|Q4HsDWBpn+4)3zQZULK7reP3;KT99p_Zp4!@6Y_B'
            b')6vz{_zIZntH$xkkK+&i)zQcC*$7@VX;TA2n|KkVPktEZhfURpmv$&-&B@VmPrtpo9$gcxz6SRG<Qkt38z>sAUijE-oP50M`ya=n'
            b'590~F#pnNi+}r=}$NirA`u;Brv3KZy_4U{4-NlE$m=9K~jsQmYETFB%52N!dxBdR1zjv_5AFi$^<I$OaF<}I}UR(6OYKw!m*n{8T'
            b'2_E`w@pW75x5Xj6!q@Pk2QA*V#XI<IztteFyPS++#sH+?LGcFUUM@2a;qLK&tybgeax}Rb`&Z-B@zM3gr05A$*&Fv@{xEld;vb)k'
            b'Pd^~aUUQwwMTqO#Y>Q?F(nDtW8}DoS^`3stf-Iz8(r_-DFcu5|t5AYg0C3>G2<`N*XCr`fdOe;Xpc~@Sf6!0-*9;fU0SW0YH}bJV'
            b'zg;xIu|WF#ETLUbeDe*gs`WLR!|Of!#TwgLFV?eCixu#^9`o$t!`MH+n4FDHtrjt$j`9y$_22H>U6-cgvH$-1z(x~<_FXa3zZz3m'
            b'w9}JoABQ+C7?w|-?iKydYku?b<42&<$IFA48Xjct;orxPRtK{);OVT{_?`Ih;mn`>*z<v%d;ZbI`Ss}J97y`9*TA>d{`;{9?djc)'
            b'gTj#5{0N#K`9J;@H<$I|{p8{r4j$OUgxK+9bT;-cj*qX#*R+wa`M$sZcHjD}_g3FO*xPexbp<?*LSMJ=XdlCF;%(S!T#k<Z%pBU='
            b'YTWPpy?6Vt7tSxvPmV^XwXOR7{YIlf2RBIDEL?1YXvAs=Nm}ny5N-I^7N66giCp7J3sN%4Npm?gic6S*tc*~7fe#>I<0S6P1Na%j'
            b'zF{4vv5-sQqs(2w7Ro?s)6wh!&1J9}@V;+}&bR!V(hZjS1cwno*`Z^6VA#&+5c*##K|_jFpKdXj8l`0#D>XwklsHX1b?KK9$RGsW'
            b't&&8|F#ZgO!-5ZG@t!fo4HgL=u-OuHst{aONWvHn23Uh6g9Gt%m?m-eRsypfUH10BQmh0=%n1sSW=m56=gwUV?4H5QnL)Dxy;S~v'
            b'69Q$_+u^X+?RWQZ>9+7NuoxL6X(W*WhL+~oO*e5d>J6g`Gw3XE9-Vn}S_q3UUT^Xt6R3DQzmxO(;c*aU(s~)JN##v8Irh0uH|nDy'
            b'j6tW%YmZ0zP9$=vf{J4*{|1TxOe`p^VZ&mrV!SM009u-{m7chXSyS}V;6d?p8~SYGd;s&#+X8h@`juEcUUyMZNP{QC?|2?V=a%>a'
            b'wm$U~f1+M+Hok3%FTWJ;cAFn7vUIqDts$cY+6JU7KLFF+3O>7<7A38*elOmUy#korbb3?q?YAXlph4@`-~qDXXPG9TH7^#Rd@D3~'
            b'oq@A}FmkodpLo!=SmaM@IfS=Bx!xf+UZN@odg3Ge1PtO^$*nEClI1d!2#?z_?&d2lPg`Vn;S!rH!q2em9(Msvr4SYmNs8Q#w7@RY'
            b'f|Hj716Br(X0vjx{GiVNEP7+(DIVnS3M^^XH-zI9B|h~5k`rnz@dp9=5OzWAfp0|5VM*?;84)%$C@0cab_}ZO{z~Qn+G!rYtQWR8'
            b'1#Z)V3H1*8uV3N}_7E5&D6ruejpv|3FVyNT1F~5_N0Km0?sU`J>rbt(8GNzZz1iMcrsMvg*E^W9e)+bg#A}d|L4j~;d-e_nz5c%H'
            b'd06zMUch7zcVK*Y7OZbU3GTOT=pAu*V83W}gA94pV;)u9qx8e)aOuM=_P+*PaDW3F%%*^IUnUy#;13AfyMK{Z2YUlpO*P|Qt7aJB'
            b'fI^7Fryzdv^s;;_zUuXhJ%NWjH&BH8Ms)PXto;71sQ!FU)YYJWflSfNO=r4XWWp#OVv($ZFwTqw&o&Y6^kR;RJOjgo`0tHoMNOpU'
            b'H(|fo>gEaCCM~NaPp<NlTKmdx_tme>YV`+G_n0HuLip3cVDHm60v7kxR0GvTqEiaa@5N>gZ)*+)C1ZGOTZ;qBIei4tf+uC>feezV'
            b'mMcbIoUkkspep|{;{ygpLiHTSodvTHj^HPe%<$l)^9Hn~pJgo0j`k0Q_7jl)XwiYjLitqI;JauivHF<{Y6XXMkmlJ#nBRHE2C#XI'
            b'XGhK<d|m~&;k+%7H0V$f<eF%UJeWn2rrJhI$n>*Zw8b6J(^p4)CRah6hvr$ki0BbmZVOelQBgZo6Ji9AoIXsvst--?jWsSjxC6rr'
            b'XJkz_nin!oVF&@4)HLyZ51-i)gT#Xa)w2zrwkTUWJ8o1EM4L7GQefQy1kX>g78`6=e?|M`Y;=Boa^#<luSXw7*CU?|qEEenp*J3W'
            b'Vvmj$;8Z<rR=pLtWfa_Yh4?<ckCTU(^^`0RBUE>nn+(tTgnE;%37>+e4EA~$b>>lmK3C+xAXd<{ZG+5HU@;s55_xbBw3d$`CZZ?y'
            b'KL*eU5acg;vIw5Kn)n6x7}}y!q=h8+4B$VAe<LX-<E!!H6PR~@KEL?koSZ?_a23Yh&I|{e1JK-Ug|rQ0R>ug`LNzS7@{RcV5ao5T'
            b'4|f~Ir~+GDU{th9cBdvZZ&t?9aKd44RJVcdZ*PmK4fxH_`ob<$O}ZCBx<oF6b-%K4q*mcY$j)*xV8@t6l%YF0TP!oyqC{Io3x@1H'
            b'LMP5O?9>*k^0<+QL*Y0rKu*gl3)}+AOQMWp0*wBsagBMrD_fDkuQQ01af?XU(waP*z2=VWg`<4D935#lL7@UL6nEXxUyLeQg)_os'
            b'jh3^h_1)ZMx24<BpZop~A1_YF#s_nGIyxWswP>T`kwg-M5H!g9F|6lU2^KuFJ9uVmDVGKy3~oU_0i9s4f)IpJLGM9vx9J~{pro@l'
            b'QvwfUo8ml>hf=b5f&-bD&xk2Y5+T+`5FL#slauj8AwIBdbC)dh`#$KMpiIg%lEG)NK%yW91;=h@lV&pc${$Tmu0Q(kM^`6TTNDA~'
            b'XT1h=K5mvvSQSC8RmDccL<$zRO+(NWU`8Of));u`Sj53?QbSQqOC#}Lq$nIOmm5G(#WJ8BIUE=XMt-575nV#2atH83z5<5K`3M63'
            b'=@8b`juuM7dHjT@(ybIT86^+UTfqY1d4~q>0>=MN#xeyIgXlIPgxi8F=^RYpY!&7?1rKXh7Kv_F)}nr~tQ-YN)31CYhtFhJ0TFFn'
            b'I%^MtLBGdt_P0hIx72m0<(3aW_JI@MkB&f|jgOB{j!wqs*UU7ueTuLZGXqk75oUi&fFW&e0RkZB)1;_jS7<F27CBe~_hK7T1!i)I'
            b'@9P<cxr8+eS{|GiiKl{Py5a=3#H~y_NgTmKFXZQNE<4K<HX`fMY1KmEe61~5o_Kk^7LR?P;|v73MTwo4jeEah*~%U2H#r<Y==9b?'
            b'<mCL*7(+RqdbSEU0(l6`IC>9q()MnD>VchI>{+&4a7LiD!YZ<%QtT2?8cu**OcLC?s0F(FS4W3I9%6J46hYAVZqZ1D#aB^e1#O<d'
            b'%K)8Eur~vkdM8UFTpTT`qacSZErs_$5_W>WoAp<63SC`+0=Nj5OR#5v9~to$$Fq%0*%s^rILyRn;L@Z6Y!c$&sF*RoXb+E~2L>7T'
            b'`eY5lkHW<DU;}iKBCKmTD|NFsKrcuWROmdC%~s)JWa|?FJ==eOa9G{nF<c$yK<%-NI%t)(=Zfpge%`(1;5I0YGHp4K-Z%r@lo}7F'
            b'CAYVXDkt*=Mlb7QExT(knyFHWnguWkh4dBAWUsK8jA*jWCKi&*b?>lnm!$Gm`RV9{Vt2TA7vA0pI=S1}Cl+uHDUkHG1>8ECA%d3e'
            b'f<{rRbk^lhS~4$`dGySghJmF`tZ)E0aNLu1#qQO$OjjF45X_o@-KhKoY7QW*i)kZ!uuwkk_4yPgv_O`32%Hk@LHtA;5Vjm`@p9a|'
            b'0LlmR0;GxB)yjSZ32xhfY=OgXPlc9MCE6V`&nzUZU%@Y1%4+s=58Zky{ewSxPOF{uPLNj-EVy#iYPz>k*eqFp(6aAk?a<&~Ru8u('
            b'P^TYO%mGv&;0rTNF?6BXb^;g*S08$3&9;DF&h{tGR;#eP9VpwVmEo$5o>>7Twdq|Pao3l)W%rmNZuY`^q1Hg1WbSUSjBr}GSEa$Y'
            b'n6|}M&W0-l?R**2bt_6*myPo)BBn?M?cWoe2b68wh9up!3yozxPOIW1Dxz3L7v-jKn*9=Qg#yrQ8JrNCLZ@2yN(WDY!&F<IDqSlN'
            b'!U#Io<g=8Hqyo&K!b6~8wuPMohAds5A?7)V6zgoIb$OKAMY@R0hKGB5ps|K40DxJBLpuUcN}n>YTji5ng)QoG<J5?5cg1;LZ91B;'
            b'@)LfAYr?E;ne~)sSnmnrU}LwhOZ2vK^B6bTV5(mY%)4pHd49J(sC1rVP%huBBYC4dd(24UpH}XJ<yyyshmT6iBqd=>Qvt18Bq?mL'
            b'y*C`=?)<%{0<c-FN4=bg^SnphWZ}HRqi)itjB_2mwlK{<6|u4{EOu~~S7ZK0WSf<zAQv&N9g*S~yqRTNa8g<ylhR)A{>BBd9iP9k'
            b'U$qRLR~mE9rmmm0l2Rt1u5BuPCS{hT$-EP+f$<k!ooAD~S;}_dFy&8bdtqd5LBgTJp~geq>`e`+)SKRPH@XzuUo0+admw=T?M$pT'
            b'VDQag>tK?BqL^IGBqbv}i&KB<*fypUF-s}GYj<3Wu`5*p<qkGfyBSm_noWWJ1*A7t`^x6(Sg}-!`f2Y^9cSpXZ4X=$A@W3>awaOX'
            b'TcLM6t5>~5LG680dG*g~R<0@6Li?pQscuklob6<VT6SY$!#cx$TR*DlfPN^|;iewPsV$bbQ^&Vr>%~-lfq8v1s70jC#(oa<7Yyy+'
            b'$i{Y9#CUM3m_z3fa>BI=ELp*zVpbLBe(8{BUF=sJ^PEA?NqK)n;}YG5*3FUi-BU(xWyS9q7<V#O$`qK^bqD=+axNVM_2+nTI#+bv'
            b';e4(cu;zruXs3Qhu}lhq?si(15I5&u4F)U%$*#9f%NLG9)&93R3rlLfTE`)?hgX|2+ae^DM*@&kI63-r)n2G4723=+<|<et`P&AY'
            b'n*khqv=eJHmFvM4!p4A8**5<#hyQQN;SvLENmRoJdsF+VH+@OA{-?QZ=CKEmQi5A5kNv+kD4>6qi)+ldaI)3`6(u%}hn~_h)Nt-~'
            b'JIa0oO>uyGN+k!XtX-7?y#SlsO5SA|f>%TaU^yfKH~Wkhs@v|R8I<<xCGygs6ugd-nL0$==yd5lCq0y2=E=Q`eJAB!Ioyz7B?78h'
            b'-NDEOB{cbfGoN6l3kMY?r+{Jk$SZ#|ci$|$)QP9B=oMI!B}N0|#VWj9W^NIbJwZjXHJz)lPTT3!--yH4np6~SDq=@UKcLv5Psv4?'
            b'SL_5wFv-^qsklU|?^TMVIC=lYfXWhg;Jwc2Q9UQMLkl5vXk{ht+YetmU;5TV&wlVMq}`yB`tFjX6n84;GLvJfw8&QCGAR<nO15ZK'
            b'rPY@-on0<?u-?eTLl=vaBl*9_5r{c9w{DOX_t6HhQr49$b!BY;OJ-jXQ#aNIm~ULXzQ;E``wib@8&t2P)s8N@4Qdxu-v@JC694#@'
            b'tC}4eoeRY-Dl=QR@QU?feX_{WLXzBIziG)+!7o+fKDQr`BW~+<fK;I#;1l&u<=DFE{LkJdu{eZpIRtasVrYr4!i<8=?ryU$LB1g*'
            b'VZ|}0xY_Jt-h$^$!=_F_mR8B&P^N|>ers$EZE_53S;HM|EvC4Ix+5j<UYNOFRJ!QS35N3|&Vw*EE_KlM!_@}U<qlx_WiP>-t;ifO'
            b'm%XhSXO!>?&}-$qg_=6m%DIULwim{}g8N2MtU~!O&7+(>So-W9@+BTG5unOycs?rRXoC*M1A9#Ck{15k3LdpfI^qi<q}PX!o>{M@'
            b'5yrG!<>1jk{DWZqYulA?R?(j74)G<IPSYa%Q(m3|j1z^zR?0|KEc%NWQ`-n=?5P;ZbD<~~YmLZj15pZ!?*9J)sNXOERpMlY#WM{2'
            b'npjlnJm7i+q~4bms48noKCbx+B(tqmxBzo|`sRv?Dvwdf6{ncNpM+H=p<QZ&8SU+g%PmSbGZ<)9H+XPPBMYO~sc==+pVq2SQ^I>^'
            b'Xu9E*D5U>vgHA!x#12Rk<12{GZ<hUL%Z^MSmU!taTf*s+OOK#Y&OXCM_f>cAS0PJ)6td19)HmJmE>%V*rYq{5>LoOnLyS#6vs%_F'
            b'wcOLdzJh5ffUPc|c^9muXG3k<K%E^`D!B4Xw@!&d$|Qej3%NXR-E{-OFl?H#9ZIK`i)JxTRBMN27B*`zq@Tr0TCpMhEMBrhfSXzu'
            b'@QCg9`z=@rtAp0;Fvrw4%C)+b;;H}pm{>KSDm%2N>-`7KX7efsT^*B^DD9AQtYH~|Mb(MwWS)Wg3d`ksM(92(UWZXdfx3SU*T?XG'
            b'2OW}z>V=a?quzCq*io3}9&PxW=DIoEJ%L2u%Ao+BQY1|l)(V4PsaRVHG~`8s9ij(Ea(YeOB7N-XNLIytHnYXCS!Dr&9n%AphpTOQ'
            b'NYE^K>{VnOaSq(DHcy~slhh>4xoB0v*$zciL)xBXp);D9=u_Tl<4jJaKiE-+j*9jXgjQjOx%02gdMb|lRNsohizy$BYrI_Zz*vIY'
            b'cVs4`AKKMz!w6%;L%U|ncFWe*8>Wp>y;J+bPKnU3xn1*`^TM)V6n(cvU@5|2;k${k;+T(~NvEYc+;lUUEcL0ZV7w05N`uRw`<t-R'
            b'PW#c?6pp#vYA7!Ss!YnK^uMJAA%20X^n;p6fkS5thlZ1w$Rt25!-nyTEDdv-!YY#wwzN9t`%u^=t&9i%W?-W*Ej-H>#T|uI!D|&>'
            b'->ird-8l1^+UgcJ-EEGP;lcBp9339$iB**Ls5(5oihR~6nGFr2ZKti{XU%=ieLns~{8;37S`PUkPs{p<R2tGxTeCoi{Ln5u*0v%q'
            b'u6KT@U)btO$w@yf-ngwCZ~f4%H0QW!Gg2sgj%v^<FTzx5B~FdM>?z|t2ETMsR{&iylWC6kVSxswZ>vJpC22-v4Ea8d8dk%;q9=9N'
            b'2JZqk%JHb}7QEf#Ac}0eR^zr#*500(q)`rM8`%5689=koVX|=oJZ-TG9$^NUJUy2bF+%TAU9A@l0zQN*REBn}x1n(nS+Ysz(quDX'
            b'9EJItOkJJVw78XQBj7!;0`V@X?b@)JB}r7Yzv&I8ShYmArROiK=Rv18En4G~)0|#>;r6$hw?!Ggdi90-sx%6}DEHeMi($DnA5hQU'
            b'MA6pd?e((C%veP+Km?(}IV2pq^JM+x+4UrD_&fTU_lz4`Lo6!O^xUB=<MlShYf3{kjy-_+s#k-X6tpiIXu7Bty}?!kLvjtoqOQ}d'
            b'1j-_&Lk((g3m&9}r|CU(XDj|73Nq=p_4OsJ;t=Ynh>}|h!P;95mU14VTd2OC(%-ec#~d(^!nIT2B!!NiWjymR79Uu)nHkQqi#z-M'
            b'w}+N7jE`@XbaV^C=;9}A0|Gh*wJJ;S%VIRQ7ynZ3a(LhX@0fP52ww0QW<y}Vdnwl-W-=9Urt&9-&Y^ZA$j7zEovWIBIdoriF?r{i'
            b'0o3ZceYWQd>;5~@>2$<mvGUXOoj4y|pL`mN<H>j|jxHt>@=gkNQ=VL$W80no0qASGhm@wO{AVWK_PTF>k9Pp=i^f_r!27po0F!ws'
            b'L9$*S2oEuvFJP8u<I&aklktZZUK#?}Y&Av6gG>pAI8yhSj>w{>>MAM@!grRgKAv1&Vr=W;{P^TM!M8%G))p11!0HiL>TQ9r5y61$'
            b'++}!6mkAeXxI~2wh!t=T@_uBws*ACfa<N$hK(Os2yrrdMPemgDoNZ?Co{E2XO!Gb{78Jpv`hah{@AjDn3z<E|^Sd;OL)6B#5r+7N'
            b'IXNfD+-GebAY(ZVyv&%w2@351{a8HU9H8|>khQuhvYWp1&5HQ<FkR0)(oP0-z&I|>G3=E)Ye-}n9bJDvIz9ahir6;F0KQty>j=aL'
            b'wV=ydzz<=52c`pjR`o;l9W`+a>Mt(fr_s^(7vEpO+MO_0#8kmn%r;@Ppjl;YRe3~}i{+YCQG~aN=lA+vw`QixCm1odFop$F5785?'
            b'9@ZKWcqVl?UT9DPrJU6t%;X~_SVz({aI*^P<|@c687&FqrE=rl1)nkU2n>jqwm@mVqpPJeELo?3(@k1^oo|jM0uUjX5f@knjZp1`'
            b'EQi5T?HrJJAr;%p5SN&^9X_c^a0rA}Jd%njiN;96w&t5K<0iR8`UPN5fe{Pa49*lxU?kRoW-A$Fn^YAiLTnqVb`)(Vg=tr_i7jzR'
            b'!W10U^?md@`M$|40RH2ZEZESa3u%0qILSjt)-qChCEfPN*zSrMBuCBbKSY9fKZKclmJ<|9p5u-Er!UzXFc3feg!;PATvy=q(@$zf'
            b'?Be0!<3vQ^EJXznwv@ua!=24MSroM4vNZ8Bf)xObN0q0cjeVD3X&l@wNi@F$K5Eh8OhzYI+nC0o0kKvekd6apvkk2s$qu|W=~fgR'
            b'y+3K=>R=z_TL5q0t}gCBvpn0RWK8h-T|K~ZCJ)G@w70<sOfcn+(xM)c4e%CFoP@Goia4QEj5*%Ec9}qzeU=s_wViMbZ;$j#tF@wh'
            b'b*IX;(TQS8A*yV{JXORBxN)VMO>r(av3c3SSOa7}Rz@`+Gi0l=4arK*n%uvYb}j2Diig`KtF18HbUke>R?u!(DFj@s3xjg={05|S'
            b'Tb1vcPRmKlbOpD6$Y!_7Qzh?lkrZGI8RJ@@bEHUHR`uu;K)qotma+qho#w^_)NBRx-z)-OAHN}Ml7Q$MW{b7FmBA`e(Ft5O*oA3S'
            b'%5ac!qzp;e#UGwoXQo>hr4Z`EO~j{Zhi#ri)H7K{=>4UXgK{BqkCwJ;yb5i-lvd(3dYJiTmsk@Udj_a3M4>ta8C4S09YDF?tbKax'
            b'+3l3cJ*=_$TkJ_Q*mTFAMyLMA((@16<)w){tdYkq6i~sdz-UQWRZ(}+^O5=d)06j;(c~{~TAk{+;}w16VcKmdp-g&9-4KPZ+9`zZ'
            b';m-2Zz^yiK(5~$RQa(tgKW_C>PC70L8kMv(1?u?dlL<r2lHGR~G7si=UaLD_Z{R2W=>^NI#1p|^Ct0YI%}E;%a#fbqD*&4CUZ>Y`'
            b'%u<b@PJ(wFxZB=%+QXdV#>luRV0lH&YTHhCx2^lVWhJH9jD;TU@D}y=q5i8FX!Q5LI#hI+FP1N~U};5&8(~9UGtaIuTPIaKeo^43'
            b'eGe!z2q^nwT*CEU+&xucS%w$EDO~zW6dh_gzD+aZb!7-#bl1g54}bnEFx!-Z#tkb#!_KP0Rbs{~Rnp}ez8cdU&;}acP-hdKx9H%`'
            b'y%b;Aq@;*A3*krMy)+J@azFx1!AI=C2h<w>_D3o?rNmcXJ;JapszKl-Mr^$uG|}yS*sH?wW9?0GQ&h&P)ZCiTquJi3^mf#0O9Gg<'
            b'wmyP~&-geIvz9Fh+=$jz0IvjCJKrj&@`9AFj@2Dhzjpn1F9q)km3$^|!<ZM-(|EgWkxxnJwS^K>mOLuSrxx<E1cFP@aHVI-%GNfS'
            b'^i?eisiM5->Nh@nqddi|ZOw#jMx<gX@o?u(6h={Zz^YFGi}^U$0sk~Y0iYYCw}tnp<jm6kCWk;Wz*6m7S6RyJaO2Wyx&_${oK?bl'
            b'Q((PAeAb)??hGl6N9Qkk7`yEBx>3`Z)<aQeMt8Gb;ObT)NG?TOF3|E|`>=D-K(KO&4l3&k*U!Z+n>(y>lvbisjQs2l#F&VGdhy-K'
            b'H6{Wcom`z<obM6<FAmbz6UCaruOJI3<2kZkv7>p@(fev_?15_<C>E>97I;P_sX_8-HA}V<1s2b*8BB$hPCrc^Jep4hL0;aT1-N8J'
            b'm3$ahTVqwxgT;<5^>50OZ)#3ZOR>`sUpTy8pcpGA>UhVcWpvb`_`>;DC+9&=RZ`vouZ3f^u3J*w6e22hCs=1?BT#g-aj0woWz{<V'
            b'DGv487dZ+6V?qNgseIN+n}cl`bSEGAH_y_&EJ(OB>CbIgFpAAu-J6+m;VZL5_bVyLJOM*EvsqiuLiLrX@!7qD_e&HBt)^7O6XUyj'
            b'=%$s1*R(E&;U~Hmwlz+Qw@?h5LOZ?V?AKN=*XoD%&pW8~rItoxmx^rY?WJ;$&|8LBzidvG#OtbdK2skIgN%=Ry~c}0zM#c>JmY%g'
            b')a!#ynBI#(^gHA};<9QB*(s`VabY#|%n(FT<S-S*6dGLAluB%)uB=LOlT)2D9x|(j7-yh=$vHAy3yvmB^;kpl`ocjAFrD&l?0iR0'
            b'c#6JN^cYfgHVQU$mN+Nv%r>rAzL6C?p@*Mq$^jLht5U{f>^GjNbIVN*8uhhmY2>1g=pQqu=3ci^S8A0ZziYf$*VL3RwMu$+lpl>t'
            b'mo{Fg5=uk$_8K*X#rgBGxm8(Hpbg1ebMvm*Y>uh;kp6Q56hq+pa<nR%;T&F59`I{gXm*sy6|U@0KdGvegFan^!iOF{{Pa`7pYpSe'
            b'JL+yeFz7OTBpxTo9DTsbe#v5i>;~sMa0;<I=;VOQTy}-3_=;b2Ie9cd=|Egj1z8MhVQk?cNEcK^*~Cm77S5AAhz7))sNHt*G+bw@'
            b'Tp)>Eq5q6z7|){(`HuDo+n(#N2F*WK*=Af}?hoSMoE&-=L`yaeV!Th+*O>==-OTz!IoP5)fq|D%nPEj{JJ%}1Zqw?1dK9%DLj~w&'
            b'fH+OvCOx+awlEnOI>1jYi<3exFk%30NrsZd<O(&IH}kdi_H=yy-StO!`>y})z+@g(6-9iWUrn`N!;XtmVN6p#Hh)M|0ZOKzCv9GD'
            b'880=R_IR77G=)ZSU6Gr-5#{;VxqQ6x-(Zj3>xP<}U$ouwBoWyidaRc!N@iw7L~6dw5riq-ldR?n#gJPIV=?C~Cs`WSEVr#=H|$sD'
            b'B!}6k!QM-(rWhONF{L6%*lnBd_*&Ed0Heki@&'
        ),
    ),
    'residual_pre_dx2.py': (
        'e62489099c6d6d236bbb946ccd5fc9f55e75696dd74c0a1e0ebeece093bede5e',
        (
            b'c-rkfYjfK+w%_$DQ0;7&T1%=WJ5H(_b*|zlwP*dhvYS49H4G&|HrEuXC8^lXK7RW>4*&!x%9h*Cez>!lPAZbX!NI}7c>}4}>lZ<q'
            b'NfE>|F_Y6|CPjWL#XNkFGx4Fje<^}=dK-R*S4p}Ea<R(b!{iaZ9klE9daX83lZEj8`6^$f()UHUSSD!>0OKSN@-T_BT1~yX4YJ!P'
            b'oao2s%OcRflT820@^m%L&C_bKOp~e1%=bK8Nd5auxSWTPWE^He9!#Sk%VegJFt519GRSd!{q+JK4Q}yjv3wLkCgNqSmZy)snt(s*'
            b'<0MV;D6Gkcsa)pbm|hOkG)a3x{6U-~(;x!iK*!c3nBIv+@F+4F$!Si+T!wKBL_0h=7Ga#_GMKdqI71bEahAkV5wblAbA$f7FuwqJ'
            b'VSGb_HDQeDot*iRjBoPWrdY-J!q37RfUo*bgES3g>ZcOFX-L#=o(2oK>oaCa>Z_)@W3pJvIHM)=lg~1ZU>2&oNBEu&zCS)h5*?nO'
            b'UJTDJ2Uo}EXFk1cQiE(6g}DXag4nDz)HpKE(`5Mw<H`9voQCl12WiJg-kZ7u9u6)#HvGe)%OLF>Vh1J+eP#BASuH#3qovb>eGU8!'
            b'F#0UtQ}eXkPR8&O=yxx}o7=n!tC4E*BemFo#k|w??}Iq=C&}#5`GRjP1WSuzIF}pi*Q#-Nn{{;y*o-m_gDCtWeR$ug)y{r6^$$-5'
            b'mzTp!V5(QO563?Z-}{$`N5j(ryr|Qr2834eEJz>yFwPI^su3@3Urg$w!w(((_VQ|QMX<UW*t_E^d_Jh5Xs~+WW4(5KblLTfhJ*LR'
            b'5xvFd|9#lq``?GXj{5rUXNK50@LzrVt$KI<{%7;SYSj|J=$-|%QTuUlcImd?JMee+clpER)o3_4_0LC)fY)h?&a0-_Z;D;`1yAtM'
            b'ZHjN3Vy`I<;1#}x7aeHvx+&hkuf0Z%xb9*!gc$>nf(OM8$h}-<9>U$_{~C?j<;7rhIrJ}wC&R<5^HI?gsIoik!2DtE0LA}sJUn@i'
            b'D0}r~DrX_CYrQGz8AuPA;V<6T^z)AXo&;G)Khkh2>o67!0IN`fRse9|J`3&Sm(u~jIk_5+5YRPo>EG#Z{8JBS^&Sc7HaGIIL$_Hp'
            b'z_CF3e38(uBi_D+RkglGQ+T~=zgS{B>&0?XYOw&G*JGZZzaRQ%=cCiXiPa(o)KUIHtM2PPyX)L^{NTU4+PBdJp?zD7^v~KD7VYHt'
            b'%Euv23WjCP)4rttdG)tPM@K-Vql^9L8t!NB;Lp*K)xqoxcsi}u{vh7JKlMjHb$npwj(>Q5b~QLY1CoC1)bOpf|9<K~dwRF!pfDsh'
            b'KZNFo{!hQ)=CWSA8=YT);(<L(h<z9hPKW;ahYy#-E80lde9zx|y=Q&ad#mg3@9sLZx&$6ap|4wbw2xso@iuJKE(V8xVGixAHSTu('
            b'&YL~h3uotN$A^QHjjg)fy;`kCiW{VD7S2{dG+;A?B(3)`h*tcw!Fn1jkt@`+ASIKWESCdgxP%$V$_ULD_y7_%PU6-yfWJf7H*CYy'
            b'W^yikl(`GoLK#SHQq3OFTn39C@B4;meaF9)Z7|mw97X_Rhtzn_u${3X^q*9Mg%qnk?P4%BO2af(W`=4gahj-g=|>4<5Q6PiNuqif'
            b'e+K0+W2G$KF{Ze|BEbXJ8-lb7!DWLaj6pHL8YCGg#Lr=x#O)gi%yxLu*?Xl}2~^B63Xyt4Qvv7BT?_1<z|5IJy#l>d{%sWkWz^e#'
            b'ztirvcX8>~@G!6#86;^WkpcRa<=9SFaWU#OqY5)<&2S#AX?<J>i!ffU@;(!&cssq7)4TqMAj+imGFXzyo2+u|bCs^tM?)BcPMg;r'
            b'Rr*#Wa;bubV=Dg&h5$?~D6L`5Vy$AlES~{dnz5CgxQf|Q^wQv7@pKdVtm3=}^Uj+BZBP11tR64hXegw?qv3bd$I!VUzJje!J;k4?'
            b'7wE><4e`x4;>~vRV?&mdEBG2RnxSt%#_~Nd-Hl+~RktW<jrB+IhWr)4<hs=vi|@WGAp;E>zXcDF4PRuMfYm&kf$^=-;B^Mh0>a3}'
            b'GJoVjn_`wfE@dCy0_8df+<1<r9N38u@Hb!(-$-t4;g!tinM8Qpj&V0%ae0~|yA9{qWEOsgW%sxXSSp3ExKC2#cBBP%85f*9Cm66Y'
            b'pqkCfx$=WL|D))QjmN0S-xOHVtgi{jF-m;u10*NZ8scjK_7HYK?167Y$6-nCt{D+FH5ez-SAGnd>i$CJ0s3hkzuYWraSGg~0Tb%%'
            b'cVE858SEi&Mo?gb8jYu5LeJFd&I9sUz($fVOYU^t+3k+4uNi!?+wH7xEz@zY*Xit!*}i<;P~tVn$e}<u8+&&4d!6o{>UmJ~q+Y;e'
            b'_qX7DcowYhzzFU(ZRjm=uy4O;w1W(J)MFl1+@tKnr*P@QEcU(yTyTJW8_cSJb5|xB^xzK&+q-*~R{Og>SWPwKPGiF`zyXC2hfhKL'
            b'=;>woPQ2=Li#>r#o*O8_y%jCJF)P1+C#pZ+5p^}_Um;U8bCXV&vrHJnL(Gyz5XPC2;Mpp|onFjQk*8;v5dZ9GR@6kQzYV+fMmtZ~'
            b'H)&Wcd2*GX)Y?~myQ_X}R;$|^yK0VP3*k?S!QQ7k0v7khR0GpRVo(b7_iQzVw;L3Lk}*8CZN-7-oIZePK~0%?AcJJ8<%+QvM=Xm3'
            b'n96_3Si!(ZXrAM^lVB2p3VsyH1Qjpo8?ctX$XJ{n?i~mnCm{RLq63YE@~LdWx6w~x^D`IJ3Y2t^=GlFi-+IOeuz8GUN5LU{UIaJc'
            b'v?-7@7*G)unrMnVm_(AM+C)mo^s}5b#Vyd&S1LY}iy+QJ^Q=Qe^aw1shN{}AXdS8vF#<?VA0}Saho-k<jSCO%!0^HuS(A<Cg^W`;'
            b'LO>=pO?=<MXAZ<5@j#(^w!_mDWp8K8jS7NjwZvEotUG|<`6<?7hwbtg^iNI)XIICE{^{^)@P2SL@Yx~y*y$O1qw*8Gq*j1a^|aaa'
            b'R^aAQaMKpzhxjf|?qjx7vOJ8?+?}s7)b$DVCR-Ce1&<l*^)PBpqXc8F$bmtuplRO*nWw;_-v=b};0|alA3#h*j~sstpb;R*U-D!Y'
            b'JhnCQ3+^$r#h^$7N$wfIUyFYsDMrJ~;pii{cYirM|M851LDX;&#@^Np2b%-X+-`-m4`Wux2-LzfEVyz<e0zZMy4Z)?jbcoJH7+nF'
            b'S|z(v6Ph<0<LEfyu-B?vPxrUC#n=YC)3?5G2vw8rS&%M~%V6EF>>R08cowp=TnyN8W)WrRPQez-jkPGzRM~<)M~^Uwa|JuK!KOTJ'
            b'<bGc`K?{)6vdROufbx<k=a>LvJZiYXJ>HhBNZ@aBh?R4TNZ8VvBAcE1mh6S299;|!b(o+q0T_zA?iepdldK{b;j%`<S=7zl+-0|='
            b'+u>ij{*OoJCqom1xi}e|4ZB*jk$NPN#2^F>@@@$0IaGoLb#@DNww7{f0K(t~<P*>d_9_TL7!~Xu6nCrcJ_$<FwV4ulAlnq@fjpFw'
            b'#S<uGVm>3LEJ=h|8$on97>$mHBZYX+^37ecOz*m2cY-k~(?|xN!2^kc7z`Y{olTn2=#@Vh9bX;!?*^C0munOO=V!SDbdFZ@IjoAH'
            b'(5hl1Vj=|#+omB{3NRy(TT4tlv@GJ_HrYT?O-m#3pQR|Mmx~o3sB#(5jsgyh1S3CF(1<P}Q@I8BAzuMQ=B$E%e_Dh!wWEcSppPF>'
            b'E8R#jkx_CFy%j7Fp10`W&S3oSWh_%bF^FywLbxf&lg_~f&K6;wQ}S@b$|BLt%39PfmX)(0Y5K}1a`;Skl@QU+rM;%LO&n3xcr!PE'
            b'q{8T^=C<Rrk3)?4f9%+{!T>f7U_E#T8_EuYe(ZolRxB=?xQH>BH^3Hb(H7(c?Q#qR;JM-=qq*QNv`!<!#=$fYSPWnq+@LoBhN?<8'
            b'DrnOLUIrMYfJF>o+N~^!klcDtM?ns2FTt;;tvJHpsgh%wp)n>BD1ftYJ_o}$&K2SfPJ9~XTbBSmkb2@Xuv5|kmI`rj)ZokkIuxMj'
            b'fzjyQF1ZcxSK$kJuyxx=5mrjhN?q^v&|R*Bt(8Zz-Y8;G?4}@~C;RX94yyY*9*)BtXl9gA2lHd20CIz#Psc$V+<K)^#tjG3jx*48'
            b'sc~;yijT^uax!0Gnrm}VetR{inJSg2f`CaV_p5j&dx2@K8oKs1OS1X0b5Quzsk~7E9TH_6H}!7An_EF5zKMNe0Rl^bq_+*=*1-f3'
            b'w2WM|o*N~hm_PEMR2V^a^b^7lGZ3p-;Q(+zFOV6|ak!;S7b`^&Os~UkR1T$@0~q+lw2?j7oA}V_vbZL+Kw-29+(gb%83pkpZ9v#^'
            b'w8hH_(gG+;@EJ%GwX2n*0Xo-q7T8sTUyp@4t`gRcKVTM;*011~HLE83(!uzTsuSRkp3`b;0|n$&1PiVrdYbM{6gG2KG+OpOZ$=pW'
            b'%XY-t1U4BG6>|Vx34CFuDf%un+n@xC^74J>wB8i(<8*ISZ!`)A&4IFxS{bg|=$VyiP@CSx5qEuwTeeRw;bt$~T(t&DlexRSG8Jaw'
            b'UR5vQV%ijI$$eJ{+JzOS>smT(Q`yFENUb3ibRbA@!AQ35M38jXu@;u~IIRjzRHQqKF3O*En*AELRspCt3{HqmVFoQbCB;+VFxD=L'
            b's_V&vFoMn-3Li=;sQ}Zfa3wX&rm%~Jkfk@55_t|H#X4K{KOPkUku4&#{=x2U6S#5#0I(dbZ-<IYbv6ceqpZnQe0x(7l^W6Qt~k%z'
            b'ShpmsMD}m+Va?i>Sx<!!^`0=17>*CRL~p8ifQhH{#`;yyyc?IIthdYk$|x&lZ}QbLlGiE(#Ec~VVI|3}7%VCtR+Ut8NWzw;0#>(3'
            b'QrKX-JDdz{{jH-ib=k(yYq?_M>Cmew*LXqb)zsOTC@hBeV48m}k}7LhY!Owh#(YO)tA(c^7s-$<k>X^nnPpRO^->qBmtGzx#Rah='
            b'WWRF&t_+@68jF+0Za}V*QYQ1NeJZV!GRuWz-U+~ck7wSd5Eyr}jO`+_#-G&o!W7PoghNH-g@?M{9UD@qH=XfzbSb#MSX{LBKmq~U'
            b'nOLmA;hVtL!ECgmm|RRGWv!pYsXKOj8`FuHrBn#CJ+8&rm8O7-8yc!z_bMxr#z6l9(vH=>vbj<#mPxUB+IwRsYW8H?eV0UtJW;1y'
            b'vdHXK*d0&WRnJjShiOy+_LG*C>&i9If7w`l)vHivTX~_D-&pvt&amHY9@TWf+=c3Jy&1=`Etc17jMtK|#Z-QUdwt#8h{z@%`zh3)'
            b'akPIYAKPIO6Tzus4qa@<mANVdVPy@9Sykx$l9Fdz>^G=+F16<Btv{i2nM&t~!SePoqxhrZ_Y9mn87pH7T<cAWe!Ys5)WBvvUTEiv'
            b'u3PBm4Fhh_(3lp}?<na-xw7rFWeIVk_cma_B9Q!gOItpp3RU~xM;DgNdbN&yW)E*;ZDoUyR1pb4QsLy-&sBS2GpW#LrZE@663O2*'
            b'*j)EO?a@x$Skk%~Y$0q6I8~PMA36O0Qx2CHU`?VLKG+@GPo42|vh}~su_TWpfK)r$PzBHbvqAy=Mb54;HSd%+11d^vng~7R@6~Yb'
            b'xhyJ^0!y)ndrDQ2sq#)$D?0<9+^X_q8G>gCdf+)E0XO@M9;(~!xfzu9>jm;suaxDCl8I6xZW^=np0nenmw9q0W8bOGRslC8Sc!mY'
            b'R<|&6K?zMh;LJyu>C8bz$thr1R(a)*=8%+ym(qCpELVXgd15p$p1{FVNT%Gp90@9_a!IekI<42;z7+>Ab*U)bROF77eL$HrpQ=!>'
            b'0M$v3U^S?lQgMm4xmPKZ;uMG%11eA4f%meYNA;Z44lRVxfmOb?XFq)FeCb*b9s9wvkam+w>bpylQr@Xt^h$xL(jr?Gu4G8`E7_t^'
            b'RexQwbhbGOz;+`O4?`?YG2Oq9BM@_JZXGo$4leazrL40%>TFXFmdrj$q>eWAurRfFeTQ#4_8UI()T>@es~w$)>TO(5eQ?MLN&Nkv'
            b'u4#5)Y%UbLXw0k~*DBVF?a88;2T5{`{l=w81;11k-rT-Nfw;9Ya;Z{1z$faRD#mo{zMnjHU~veS!v|B^V(5u4!i<v5?ryVB%k2=7'
            b'u;N&JTCcbNnt%oDOv9#5L6%0z;ZUxIBYrpd8rtRP*|LT^+Ez?)TyaZE-~l3Y4yAM=nk&(!Nt_2^Y(nZ_?T3pM)>ZAp^vh9#owdvy'
            b'Fqgfp8E2I63eZdCyoH+5YULufeLD(cpViz^6ss`)OY<m~nUy}fO1{M7B?44=4Nqr<9Ba_QxMz=Po#w$mYssT#$wqu8h4k|D(G%-6'
            b'H^P{ftDHRQiN6!9f3An}%_`bc-6Fr_(rH|zf6B{KfN`==*h)F6iba1FV`?7(ojsK!c`6m<VyzK*X(CEN(cS-(fchO1P$f=QSUkbN'
            b'FUduf^Z_4SC-c5+Kvj85@?ptm<(O@)6Yf~s(=}%q)IAD?Tyajl|5;e)>6@iCSo++oI0d0>GlPLPb%T3H8(A2=%!I4*{<KwvmJ%LN'
            b'Le~w?&LI6KD+~&fCALqN7@t+EznypM4LdV|SmNoRY!2EdmmWc*;%<hE;j8xUZ$g#;DP)~JsE<V9p_`0cOjp!9*IdwC4lp<Q#A;cq'
            b')Nr+deOAvh02^&U^EOyY&xYExfjT>?bTh}V-8v--DVO}YE#&gNbtno1L%(kBcThI9oYjkYqFLK7v#?o%KK)(1q!sJa-^EJ~2yj#D'
            b'q!zK=Ubg`&VRg`!9hR*6#<*6e7d-WU9}BB`bR7%r>CN$jdcA&`gRPFmN>p~p#kjBxz@kc{I+bT&zQS_3kr6uhi03BM?J@0N!sQ3}'
            b'zk?1*L-oQbq*3qMNbD%ga*sCrb$wYMZ(l&7k8e-{PZ^S?3TuJMuT(6p0vd`U!4A=bBRRdgZjnB8bSA6f;FH<n*sSuT`j+JZD#F#Y'
            b'A|&XRJaj55j<^P{zp+lBVUyGp%(-Y)!C6m5R6|;yWMMO!nP@HVv~eb<${*~kLrZ1*2tuQ9!`%5-Ry`H!KGnx3@C3tq6B;icjyIm*'
            b'`eBOE;Kycl+c3i1aNoY%W4&c#?G4k$nBJ*<vZF-kmmI@*iN3Hr7)9T85m?GFSop4Dyqe0#$fVO!DL36rCUdQo6^xf5Tj_8abbl9C'
            b'+G;;qo5FFI8#NWBK$A(ul>S{-5aJi8D`C(QDR5}5;ZSo56PW~PW!Nx&m8W5}DXcQZU`wlGzW0TF_mPR<U-xVj#zkb=qPSy_DtL_|'
            b'>YEi=qHAY9W82)~rrXVtDm-|8qr-!JJ+X>AAgYw7S5eOzC6m5kwDq!e{H(dpIrzn&h#!mkPAeea=V@6VkxG5~+qNt)Am6udsA^i7'
            b'7dJZJ*Dq{yr4*#^7jN8FPPD%7UP<PJX)`h?e9mgnD$l}HStZV8aM@$VdklWG(N+LkGLvbJ2Pc6Brf;Ld)Fo*~WW*BU8$6c9c8r33'
            b'Nl(Z84G)dg%K51E7QEf#Ac}0iR^zrzmfo(Jq%jUBE7<$M89=koVX|@(JWa6(9$*GoJUx|^F~aE4rb|II2>1{#&=}hCVg`+i$dXk$'
            b'm8O~r^C-;U<m&3WrrC{T9{~>(6^OUVY}by>BuS#8{dK1|#_KI~TYCP=dhWG4<DxY_InC+CS8ji+c~g|(t5;vSuS%ovi*moUu^5(n'
            b'IRM((t0-EVyuDu5eI(ZH2_S;d;Or9)?P;=n^z4fVuK7F0nRiVXTSF|aFzL8MS<dTCipP}tY8-n2^Hr|~Hz{aeG|+TWEjqom28QGs'
            b'h(%qeSqYRyOotlO-WEJa3s2L#@6J|yEebN}x0^2_ux=4hsv=5mC<SY8HCW1Nh+(147w`N{`+Lj*(<odzH({jE(X*Uq9>(GW%QrK_'
            b'S$=VAulxGIa)$BowUUnR%^~`h4E6y59fMkxC-_Y<n%j$il)D`4JHT6}9V~(uJcL;v*zZotC5V|!C7kJ=3`6HohY{q%(&NrmO}^>7'
            b'FWOkV^TYsZb=^Me^M!T)gJ`u{Vm4d&Y5GQ-4X%zq4#kJja3~JXM<a?(3Jy~qpPymdt^WY%D>{dirs|%^OuX*2U;h!$0NN)~wPk<@'
            b'HPHbk_fmpnz1kNZVm6(@EKi4n%O6I=_YFKW1hCm_ijsSo5)5&u&NCg7M@`pos5}UtS-LzrzPP~L*7@0o<L_al2zZ$;Wun^*@R|X<'
            b'6KsaC5y61`+<ACIhY4qCI7fpGh!t=Tihg9dx<Fwm<!rSCfZ*FlcuGs>o{B~QI9pBNJ>3%FG0nSRSWpIwE|PoQezV6kn91xhp5CTO'
            b'9HKR@oiM~V%qciQ?mk=d02%LHz{89wXi#Ve=*Qw7=K!tmgRIe3nceh_Z&t*=gXwzak#;he114~BhH0<dSwkYr;PC2)!O6+bC}Nu^'
            b'1NaOsuOkp2%z`#s0Y8TMEw~QwSzR=t&!~wTFn@6YKMoFmIRD`i*6x_OBBrZO#bgymGn!S_R994}dqw$jqX;}dy17;ju9@lL5k`zH'
            b'jA6mlL-a_ihnG?a)JZMQ7aEknC};BrEBVL>)|oU7+^m8+xe78%Msosrp~85#!Dq}o0t4cqEijsI>1gQ$?>1Aw=_IW_&Ns!g<cJW='
            b'h;QNri%{)^EQi5T9UPE&AQjuo5SN&^9X_c^a0-N0Jd%njiN;96rskV4<0iR8`UT)mffEbc40H+>FcRxPvxN+@RjO{WL2PU4k|Eko'
            b'3e&b`6I<etgef?x?fV#Y@_kcT0Q|=*S+Jo;2hv!XIK@MU)-qCgB^?XL-0p%IBuC5ZzeIv~--nrek`ok5p5f8%$8R_q&=a3Np}p=i'
            b'*A+N@`lM#WAs!w+PDB(=QZxWzODPOI+}X^N#my&te@Q%#U<JV9QTLe8#=cGP4i(%jNi@9$K5Ee7j0VSe2`-I81LCD_KsrvCO;)sa'
            b'Bs=ifq<i(;;N5W(R|oqb-vW61baiq5ndR9kC1--y@A4j&Gr315rM(SKV1gxgR2Frgtbn(G;v|&yQp5?BVodS$waWzheqd=)Qu_%X'
            b';O(J)X|-0AukKX2Ho8zu8AO$Bn5T+d0XMD;vnkHyCN?iS7;Avc$I7W@HAB7{`;ctpEGhh3>DRK2qIkGzs@e+2O<%nHIRv}7W|a`|'
            b'txlMfo95RbrJL%Wr}4O4v`j~Ey9exct2$MR9v4Lc#*s0h1=1r$*|NI0JOR{e)?z6?kl1NzTtLlMK>yV&@U{94*^&fAmoQtr^jSHq'
            b'5)GZeWrJ;)MwJW)DQC)%gl+uGGwaNBYoioGTeyz+)a|s*Q;2$|stBXMRB}+hvD%}hZJVe<Q!k}ec#RQee%U70#KxWh+Eh5I4nam&'
            b'&gl+d+^?2Cz4h#N%H{5F@cA1YNi*2AhaU$g{!!`qM;-FgMDA~p$1W5w!K=V%NmzCD>Zs!*^Z6&o??!{s&u&?r>bT_*edJ-<Z788k'
            b'dCR6DideOCW4ebs%Toh4n!G{Vb___xAesKS)k`($xFqOQ($W;D<D*Y53^7Y~*Ph8dnBIDg_H?;|zv+&C%dJF>;4hOb)WzmxjR(29'
            b'ztbxKn)F_)({S8Uji4@qcLKQU(RkX!T;s;bxG3N~e45qPgYIry_j|)CO0gLWBii8&+V6e+&nKYM-+gtU=rEnlpJ~C;iV!!#hN5Pk'
            b'eHrN{sp9dA0zVylK$$^6IUeH@uJi2fsY=T-ya=>#=_^sRujTkU%}mslA@J?2E=GFzv)_Q(s+2UYSqT~rRu!QVGhXS+Rle+1W10in'
            b'K;vubY|`@<9o)H>@(Zh!3=wA`{3yJWCO}k9NPsK&fE`#tE%6_TNF|q)`0A@i7&b*U2t34yt=E$#y1fs3RakMX-7#*8%3PJ2TOE4T'
            b'o9mQb&swcX0F&3&D!Bhdj1w_y*pk4_XsspiN`Q6nt#T^QNcrkq-G23JH-7hA@~$w+C-NqYc`-eWx7!x^l!RVWC^2Qpqmq1TATLWG'
            b'xC9GVdX}tgtdmI})uNOt%8Rys<8w61Q_R}bOxR>ZDwYzJJ9nZqiaG;UtpP0N<5~y2L%#sf4$_+<dQ=K#>3EYvAQ|AD-fKr$%It9M'
            b'(rUa0*-e~P!rCdY-XcG1^np7=O5>6K#Ry}YgI?Ea8sp7Sl+I|cmNQ)4N(3pSh|2|59&8^DE*c0{Ezy4E1+<&zV&AXYuL_h_qEn3g'
            b'<POA~h<|ea{qYqR0v;Y;9-p6WlK?Li>C1^?&EPkX1&r|=S+Cg9yy@tDH8%FZ8yYAU+fXg=gh{di$;ZVcSt}HnJ-ucy6*fBkG`aU^'
            b'J{1IcetQ<+k_p}9!?2nftBM{hc5JDCU6y=fqd_ghPD6Y~c|AihHcXUy$E9U-lu~>~|80`<V5cf6?||3BvD(%xnQjUZRk{-_v$7K?'
            b'TG}~OzJT&-o%j@|dhLT8g@7@s0hUxg+obj0x(vFN5B$rsY%dEE?o9f387vsZX07hcOgZzFTcZ1w3}l{wA)MK)jVGb{O4Rrq-of)F'
            b'iiB2Ey6qA3yL#xlRfbo$4u|0<Iv2J!PKu{c44Xnbz2ofHRxj7<hxShgsP&<iT5X#f*)ZCR1K_QaGQ|7KrgY`>rt2M_xE~CYj1Rk='
            b'+OwB@L5p{&<2n@7>w-_1-ibhTTNFOxYt?45RouqK7pq}ph9HWfhN-wrp~kmg(p}7GE33PkDX7jB51Dm`7+0Ww!!<H|7aUENF76B|'
            b'>I;e%U^>^kaqt}@;VH&eF=9x!vr)34HOD#WV73Xx@`<eA5hMJ3ryNl6sawjJjNRH37ou{L{o3YBP3;Rxo!j>C><))T`NC?TLc6x<'
            b'hE5*mO>M&^(fs*?IrzBY4mG5}hQo37dVNUu1L>>bDa_ATz!BjzPdMzLg49>E797iv3nc!hPwL{xUYE`);X?->K7A@!NPd=aOC6sF'
            b'9-4;_#D)Z!69jnqUNV~@M}P{3#Uzddvl(!i$~OK}ad@Tyzv!#M(MKemwLs5}GEW%(zYo$G-H2<_8V;A`NghN!Vjna>TX`BTGj*dM'
            b'Nk*anNx(3kMk@;M>=L#eH>M26d8`V;_!79U#lN_I^EQa)Y`w>L<gKfV1NcOk^^cxli|QH!UPk396~*U#ofvi-SNGGSsP*V8K-WFQ'
            b'Y3w%XxJ|HyDU#4RcxqV)bG^Wb0kkDCM?#G+k-^fIZ>_f{!?W+Nj^OQ^?wfs698`60;nV!8@3tz~aaOv!(%kTx-zVyhNT#4CZC-Df'
            b'Kr$(0Jn&MQLajJa$W3-cc|LZz9k2Xb?6LjTp5|~CZMQs0M0Sfo<+(~^nOPB$>d$ioVM<3CtGPlk<l4em%z49gkcKtOZ8xzS_A7Ie'
            b'!)(-G_c>NmjE(ac(>+DlZJqD<<k9~Ds&rn3'
        ),
    ),
    'route_b_rc64.py': (
        'c2d9759a77e793d643ca1d4a557934cdb66f39473b244f382dd9f0b8faaf89e5',
        (
            b'c-q|?YjfMU@jHJ7qE0VlBI_*MaWk&vnZ#C`%=w(uaXZ&ZJq$!a7B3X3<HL^blK<XaJV*cn^-3@8R3A(dSS)rIyNlfg-05`Ar!4w7'
            b'&*HSCqlA&S*Zsq1<of*hh>YTN9H(~$nPfRBr;H@g0lChqlI^`DIgMD)^KPJ-=5aZlu`-VKqHN4^lC!AFbCyPo#042;d7eG6F{#pM'
            b'O4B=do8;MyW4z00T0#S#)%)-vkB*PR5sf~wbli(ReCQE^GZa~s12XohWN8s+DH*XOdmvewETD0o6|5j*R<e8+r*Q#<ctuL*#WX9w'
            b'&skB;07FSjRx<*maYAW2hH>J($Vif<cO<XU0yvn_IQ62O7SlbB0)^A_I-QQ^aq7cxQk7NC!jQzXd6t)eEX`mRAg}N|*%p<HIV;px'
            b'QRY=tit%$=PLp_~2ET>RdQ@7?<_kg#lFmKPdvW>p)r*(I8!{lzyqh;SFD~J&PfkzCVc#46^mcfDGrS1j{qx6{Z!X_K*YW;7c|s1K'
            b'AHm;e{R0^C!|*En@x@<XpF{7cbA5c#?|A30hUY)Lee?S2#^^eSuHn`Bn~UMK-hMuW_KTrte>Hpo59kp5^xpA@-#I-h{>FcupvJKV'
            b'_T@`}^*mI!P@`!0)6MYe-Rn13Fff1o_%TQ(EeTf+xpbjgVBaI~i$LflxzFM;d6LcPFBR+Z_6o+qpprZReX*OS@!iyJn=`KSP4jc!'
            b'O7k+NiMur%#bsf2jcG|mQxX5lpe3Sn8pY+pXncZCFCf2ZJm^f4C+cNDd{=iMc>OaUA5Vz?NOgVwtdxB3jDh``EfX}8DcBt}K0-^v'
            b'CyE1JI4=A>lCt2Pl2`qm^UWKOjR0&0>>(w4eL)UN9O$m+giqPQC(?TGTXv}_wO-kz03;ncfM_xS1aZqyyj_AK**XY*Yof$8thw56'
            b'I@D5<Q*&q|frAFvf|#*cG++2!J(|(hCkGWMn%Td2o=|VY3WRs_O(kAdb%&-q72T=l%W5uEv0d2>wQf~tOO*}2qMU)h_5-(Oqh_5E'
            b'sjuRB#<#5S=FMAdGL+zq{HfCQ+UM*3-FHN+u%gqZH5@bWS=mA@p`=Aym8-vBUW&#qHJLSTjtZ8C<@ZdX=yJ*Y4l~w@WvP_-tLc-W'
            b'W9XG@mDlYQI%;b*tQ0yLW3??5It^R3O%MBZQ+1kV1Eg81<X#TowovZ}a{K<h_5(z2`fMO8goq=?jo#(ERXp)Ty_mkc?q2rYstXiY'
            b'zG}YPcGKh$Lhu)zqQF)e%N^H<m)vpPqq$=~h1P`+*`-gagXrz6eVAG~-G^=BS-LxTyQO7xuA+1OtR11=@HTQpT!5!UtX;h{ytNr;'
            b'i1Lh62%+2aiaaI9hgGm_{F-K2Iu+3*$s)gB39HX020#!jYg;>d;Y4t4;bHYGxer~hboXIsmEk@NrGlPDx173!acG-r)hqSI<H4dA'
            b'CA28WHLyNJ`|<xllZx@z;Dir^I3Du~p61NyBFX5uOXhhtqN6y$y*;5Nq!Fx3!1|&(L&9_>PVZ?Fk9#~7@WyOHeyJddLA7cS#H{d*'
            b'uM>#nd+C_wIb8%~?@ap?6x}0<=>DLC7QDm60Nxyvi2?LafZp=(AW(}b#sa>0Mjj2w0TM0SiYcA5_x)RJuM-LAhE$~n0n$$zw?X5A'
            b'X`<^08R?XeSp`NqrT36lA$s!us!N{Tc4~SMIkJ%?K0M+A!%f)D#eA^+;r!5@E@Trq9gzLrey{@-BS`8vcAm=#?YL)>6i7Qm@GU4+'
            b'$tx{ThQ5;F@lntN0H1z}i$TAkUl50nQQ_BE_DCOy1PeW2d(XBYc7cRQdI1qBGss*Ojgy?gE|5mCT6`c3aJ143o`L6pf!4>poInR('
            b'GvH<hOlg?Aqz{^0l>@C!z(2%k%5q&o`5hoIBr>82#aw<MW(GNEXqM!s^}R<4=9u#P2}|qs@2XjE$<u+tZpmI33u=e1G%d5EiVM-K'
            b'TN;I60h*f6z?y&MX*-$FIALR@Ji@ndbA)ZFjPI(fl7?xBwl+)LZO|%(R|p}y&uI=;45|GIPw=_ni;_U^{NnOb&?({21vXtiM1p)|'
            b'54OiFffzCf1YOnhc>~JR<>9q2KckRuof7cyU>#)bLso-okdxJrTubHj$4Bd?S4xW>Xces3vjbgPJx98LqQW5!rYOXCBb>L7lp!~&'
            b'Js$#N-&=!k+eeG1F}T^}O)On5<St*8F0okSgVxXE;18tOK1`(fJMtz6Z#yRj3p>uphc@$pqPN4M+;*IX4|TAmErolW%rG`wZq$F{'
            b'uMOH_yWMsDH69Sqk|v3`Pfm8b0W;gKuZY&y$Uqo(cYT9<==1K<x0Ba(OiEg}a0@CYkq$-Y(nZ1TS`0hPy_-vFM&r{AMK{#giOV$!'
            b'zttw`m9uWDT6PM*tAPpCUq|ZJns4T*ZsOdgxTE?TkZV-kB5n(HE3=X&VI2y&h6-X-WVULw-$Y;Qf-iZxLn=J9!NZt`9FJW%!X`2^'
            b'gD^RDo!MPcjPf~VVd^I`afGld#B?AG{emTvt{NFOL0DJI=T69IQ34sC$rVK8z&bv7l!h-2Cy1-p#A$VCr(+LHXbc(ICzt`#MgzYI'
            b'SC`@k<S#TSnqH8&nRuEc*%M(lu2I=@mQ(A|3}!NM;Q~X9-b@*gQ(~UtSefGLWme@{lZP^|(g;XVLALtM=v^Fjc`ZkjZ|eHADEZWt'
            b'X<`)%{UvfNnyYjU3K!y;o6eju%>0s7j|TX?OANQR^f(J>1?KS?q%9~{ahUK(luW)thLfDmu+}P1QY|r#s>wuF9F6@GlE#{dpog*u'
            b'1!x#$@;D;cO|$gko&IyJq;n0slIU(jVawXc!h&I4-sVE&l`)>@sWag^-K(8Plw<|tp2Oh;VE*>G+a;um`uf<F!sb3g^Jin0NO#r{'
            b'MaUn>Jg1PVpMX$El-;x3@;{svjW<D$bc@AeHp&vCD%B;c3s#m#6escncYZ9<Yq_yVA#szzijlw9Z$w~LKsG6+X|A@UWmT-yP|0_z'
            b'#Jgq>=c^~p2h}m8g}*v64onn!d16C6l@k#fgrG<3Fzi9FzuF2Wx*15FCl7HsRl@sCJiq9MP@ksMYR5OqW+qYQwIT&=gOTOlc7W#4'
            b'Xa!7taAdl(We~L;w8F`PwG~$RTWOMc#N^Z^HNj|+vxyHv3aloSi?z%~zZ8o-OtQIhq$UX%j^^4W`xYL46!;&~rM%V>Azh(bC%Qga'
            b'Tk7(z#B*_87YzQ7*P=Ow6}*0%MOm>PGOV`Z8Vr<7c^aV%56=_>C_D8o<)l6}J`H42DD@~!Mxw0|0eLF?jf@9sw<Ln)R;Q##2r90&'
            b'w(0enrOn~qu3MI6#nK3rYkcpL9&4GwO|D5tx-zgJe}#PjkjaDQGcl6nVd9FY1q9zB10`i7L!Kl_jN61`u%VBu2)KjrMV`QQbhpl1'
            b'_>n_%d+n*R8;Gbt;X+#aa%OqS!B%jE2GptZul)`<9N*u=ZB=#z0r?|23LJUm1=I4%PN#FtA_iUgUX;xs+l%q4q&jxx3rexFt4HcA'
            b'y=O@_XX2o6i}X~|CUetomW{%)%9%mDV4DT9<=fW;183H##0e>*c}Z}qlu^lZtCUfshg!P9kNS<X;;zIzcIJx~!Fs-^*jkY+=JbJI'
            b'S>b^1UCcY$Kp*kZo<c5N&Xids%(itB>5%H`B$6wPlvpSudFgLts{_6<Sa<4)lshs{Y~C*Mbln<hqo!L=y+%V?7MLY&qtJX(aI!)j'
            b'j+Uzg=q&+g)5R;ZkF0THU=3_GFBh9F*#7^(D#^l=dr?-Eqc#yo0)|=;4E%Ao`d(9TZb+T4E*&V-`PQWa>m=a6UOEu;$SrFIugM9?'
            b'MnAKtTvs%(Z@CDJi9k}fn0H_7V!}qZV7~4qjCMO=6TR5tHVpo<s+?D)V2|6cej8$`MzTp3wCdPp(@KXF%Z3#Saj@?wP$)Kx4#Dxx'
            b'Rg_&*AR&$o<7}tbd^+Gv3HpO2`?bN(i2eT;+qf0H><G?e4Azh{Qo?3TQvBx)iG4X=6J#}Z85g0@rzRoFvgE`b=H4F$OSCtL?^qdP'
            b'pbOrvVB-e8%bdTu5~S|`wE|ez0C%e4xUUKSK6L|}$sXgrqx*G=zLe{)Q+W}ch|F5wlC`yH_RxVH884Ry>@d}ew;Z(sQ)qanMZ)Ov'
            b'sS61Xy)|kCFq4)sj%7VjWO&+H4^Q%JqreRW;v5m?XwX`&C{Zd8-8rp}Izf*|M1l8jhe<1i'
        ),
    ),
}

# The JG2 source is retained as the exact located mechanism-bearing artifact,
# then deterministically made public-tree-relative at materialization time.  The
# wrapper always supplied these arguments already; requiring them removes two
# unused lab-absolute defaults without changing the executed encoder mechanism.
MATERIALIZED_SOURCE_PINS = {
    "jg2_tail_reencode.py": "6e2b72e5738eb2dfa82f2ef36be50bc3d56d7c2dfd4473f825f70b6f3802d6dc",
}

BASE_SHA256 = "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080"
BASE_BYTES = 180_456
TOKENS_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
TOKENS_BYTES = 117_964_800
FINAL_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
FINAL_BYTES = 180_002

# These are source pins, not substitutes for execution.  The files below are
# imported or installed into a private runtime copy and the real encoder then
# consumes all 600 token planes.
TREE_SOURCE_PINS = {
    "runtime/free_corrector.py": "dd337159bd84e96e767cbde9a6dffecc909e824c2f092399e09095bebaf094a5",
    "runtime/fx1_logistic_mixer_corrector.py": "8038119d065d578b6c163d2ee515e437cab273737ecf82f8c30619844c0f7452",
    "runtime/rr4_free_corrector.py": "96fd35aaf82c737a997ea41d28c2b6e83ee8b0237afcf52808ee6cdf55a874c0",
}

STAGE_PINS = {
    "fx5": ("4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841", 180_386),
    "dx2": ("976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674", 180_368),
    "gb1_pointer": ("ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4", 180_215),
    "gb1_joint": ("ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3", 180_192),
    "lb1": ("5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9", 180_083),
    "afr1": (FINAL_SHA256, FINAL_BYTES),
}
DX2_PAYLOAD_SHA256 = "b93131a52674abb4ada677e1b6cf08eebc6afb94381136d23d010e70a287e210"
DX2_PAYLOAD_BYTES = 9_811


class CompressionError(RuntimeError):
    """The public rebuild refused an unpinned input or divergent stage."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CompressionError(f"required file is absent: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def adapt_embedded_source(name: str, payload: bytes) -> bytes:
    if name != "jg2_tail_reencode.py":
        return payload
    lines = payload.decode("utf-8").splitlines(keepends=True)
    output: list[str] = []
    counts = {
        "runtime_default": 0,
        "tokens_default": 0,
        "runtime_argument": 0,
        "tokens_argument": 0,
    }
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("DEFAULT_RUNTIME_ROOT = Path("):
            output.append("DEFAULT_RUNTIME_ROOT: Path | None = None\n")
            counts["runtime_default"] += 1
            index += 1
            continue
        if line == "DEFAULT_TOKENS = Path(\n":
            output.append("DEFAULT_TOKENS: Path | None = None\n")
            counts["tokens_default"] += 1
            index += 1
            while index < len(lines) and lines[index] != ")\n":
                index += 1
            if index == len(lines):
                raise CompressionError("unterminated JG2 token-path default")
            index += 1
            continue
        if line == '    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))\n':
            output.append('    parser.add_argument("--runtime-root", required=True)\n')
            counts["runtime_argument"] += 1
            index += 1
            continue
        if line == '    parser.add_argument("--tokens", default=str(DEFAULT_TOKENS))\n':
            output.append('    parser.add_argument("--tokens", required=True)\n')
            counts["tokens_argument"] += 1
            index += 1
            continue
        output.append(line)
        index += 1
    if set(counts.values()) != {1}:
        raise CompressionError(f"JG2 public-path adaptation match counts differ: {counts}")
    return "".join(output).encode("utf-8")


def materialize_encoder_sources(store: Path) -> Path:
    """Restore and pin the located encoder sources in the retained work store."""
    root = store / "work" / "embedded_encoder_sources" / "compress_vendor"
    root.mkdir(parents=True, exist_ok=True)
    for name, (embedded_sha256, encoded) in EMBEDDED_SOURCES.items():
        payload = zlib.decompress(base64.b85decode(encoded))
        observed_embedded = hashlib.sha256(payload).hexdigest()
        if observed_embedded != embedded_sha256:
            raise CompressionError(
                f"embedded source {name} decoded as {observed_embedded}, expected {embedded_sha256}"
            )
        payload = adapt_embedded_source(name, payload)
        expected_sha256 = MATERIALIZED_SOURCE_PINS.get(name, embedded_sha256)
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected_sha256:
            raise CompressionError(
                f"materialized source {name} adapted as {observed}, expected {expected_sha256}"
            )
        destination = root / name
        if destination.is_file():
            require_pin(destination, expected_sha256, len(payload))
            continue
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        require_pin(destination, expected_sha256, len(payload))
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(EMBEDDED_SOURCES)
    if actual != expected:
        raise CompressionError(
            "embedded-source inventory differs: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise CompressionError(f"embedded-source tree contains a special path: {path}")
    return root


def require_pin(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> dict[str, Any]:
    fact = file_fact(path)
    if fact["sha256"] != expected_sha256:
        raise CompressionError(
            f"SHA-256 mismatch for {path}: {fact['sha256']} != {expected_sha256}"
        )
    if expected_bytes is not None and fact["bytes"] != expected_bytes:
        raise CompressionError(
            f"byte-count mismatch for {path}: {fact['bytes']} != {expected_bytes}"
        )
    return fact


def vendor_file(name: str) -> Path:
    if _ACTIVE_VENDOR is None:
        raise CompressionError("embedded encoder sources have not been materialized")
    return _ACTIVE_VENDOR / name


def _forbidden_state(relative: str) -> bool:
    parts = Path(relative).parts
    return any(
        part == "__pycache__"
        or part == ".DS_Store"
        or part.startswith("._")
        or part.endswith((".pyc", ".pyo"))
        for part in parts
    )


def verify_tree_manifest() -> list[dict[str, Any]]:
    manifest = HERE / "MANIFEST.sha256"
    if not manifest.is_file():
        raise CompressionError(f"source manifest is absent: {manifest}")
    declared: dict[str, str] = {}
    for line_number, raw in enumerate(manifest.read_text().splitlines(), 1):
        fields = raw.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise CompressionError(f"malformed source manifest row {line_number}")
        expected_sha256, relative = fields
        try:
            int(expected_sha256, 16)
        except ValueError as error:
            raise CompressionError(
                f"non-hex source manifest digest on row {line_number}"
            ) from error
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or ".." in relative_path.parts
            or _forbidden_state(relative)
        ):
            raise CompressionError(f"unsafe source manifest path on row {line_number}: {relative}")
        if relative in declared:
            raise CompressionError(f"duplicate source manifest path: {relative}")
        declared[relative] = expected_sha256

    actual: set[str] = set()
    for path in HERE.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise CompressionError(f"source tree contains a special path: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(HERE).as_posix()
        if _forbidden_state(relative):
            raise CompressionError(f"source tree contains forbidden hidden state: {relative}")
        if relative not in {"MANIFEST.sha256", "base_archive.zip"}:
            actual.add(relative)
    expected = set(declared)
    if actual != expected:
        raise CompressionError(
            "source-tree inventory differs from MANIFEST.sha256: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    rows = []
    for relative, expected_sha256 in sorted(declared.items()):
        fact = require_pin(HERE / relative, expected_sha256)
        fact["relative_path"] = relative
        rows.append(fact)
    return rows


def verify_sources(tree_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(tree_rows)
    for relative, expected in TREE_SOURCE_PINS.items():
        require_pin(HERE / relative, expected)
    for name, (embedded_sha256, _) in EMBEDDED_SOURCES.items():
        expected = MATERIALIZED_SOURCE_PINS.get(name, embedded_sha256)
        fact = require_pin(vendor_file(name), expected)
        fact["relative_path"] = f"compress.py::embedded/{name}"
        rows.append(fact)
    return rows


def compiler_path() -> str:
    requested = os.environ.get("CC", "cc")
    compiler = shutil.which(requested)
    if compiler is None:
        raise CompressionError(
            f"compression requires a C compiler; {requested!r} is unavailable (set CC)"
        )
    return compiler


def dependency_fingerprint() -> dict[str, Any]:
    expected_python = (3, 13, 12)
    observed_python = sys.version_info[:3]
    if platform.python_implementation() != "CPython" or observed_python != expected_python:
        raise CompressionError(
            "compression requires CPython 3.13.12; "
            f"observed {platform.python_implementation()} {'.'.join(map(str, observed_python))}"
        )
    try:
        import brotli
        import numpy
        import torch
    except ImportError as error:
        raise CompressionError(
            "compression requires Brotli, numpy, and torch"
        ) from error
    versions = {
        "brotli": importlib.metadata.version("Brotli"),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }
    expected_versions = {
        "brotli": "1.2.0",
        "numpy": "1.26.4",
        "torch": "2.12.1",
    }
    if versions != expected_versions:
        raise CompressionError(
            f"dependency versions differ: observed={versions}, expected={expected_versions}"
        )
    compiler = Path(compiler_path()).resolve()
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": versions,
        "brotli_module": file_fact(Path(brotli.__file__).resolve()),
        "python_executable": file_fact(Path(sys.executable).resolve()),
        "compiler": file_fact(compiler),
    }


def obtain_base(args: argparse.Namespace) -> Path:
    requested = Path(args.base_archive).expanduser().resolve()
    if requested.is_file():
        require_pin(requested, BASE_SHA256, BASE_BYTES)
        return requested
    raise CompressionError(
        f"pinned base archive not found at {requested}; place base_archive.zip "
        "beside compress.py or pass --base-archive"
    )


def verify_staged_runtime(
    root: Path,
    archive: Path,
    fx2_source: str,
    residual_source: str,
) -> None:
    expected: dict[str, tuple[str, int]] = {}
    for source_root_name in ("cpr1", "runtime"):
        source_root = HERE / source_root_name
        for source in source_root.rglob("*"):
            if source.is_file():
                relative = source.relative_to(HERE).as_posix()
                fact = file_fact(source)
                expected[relative] = (fact["sha256"], fact["bytes"])
    for name in ("inflate.py", "inflate.sh"):
        fact = file_fact(HERE / name)
        expected[name] = (fact["sha256"], fact["bytes"])
    for relative, source in (
        ("runtime/fx2_model_axis_corrector.py", vendor_file(fx2_source)),
        ("runtime/residual_archive.py", vendor_file(residual_source)),
    ):
        fact = file_fact(source)
        expected[relative] = (fact["sha256"], fact["bytes"])
    archive_fact = file_fact(archive)
    expected["archive.zip"] = (archive_fact["sha256"], archive_fact["bytes"])

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise CompressionError(f"staged runtime contains a special path: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _forbidden_state(relative):
            raise CompressionError(f"staged runtime contains forbidden hidden state: {relative}")
        actual.add(relative)
    if actual != set(expected):
        raise CompressionError(
            "staged-runtime inventory differs: "
            f"missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    for relative, (expected_sha256, expected_bytes) in sorted(expected.items()):
        require_pin(root / relative, expected_sha256, expected_bytes)


def stage_runtime(
    destination: Path,
    archive: Path,
    fx2_source: str,
    residual_source: str,
) -> Path:
    """Materialize one exact receiver state for the real encoder."""
    temporary = destination.with_name(destination.name + ".partial")
    if destination.exists() and not destination.is_dir():
        raise CompressionError(f"staged runtime destination is not a directory: {destination}")
    if destination.exists() and temporary.exists():
        raise CompressionError(f"unexpected sibling staging directory exists: {temporary}")
    staging = destination if destination.exists() else temporary
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        HERE / "cpr1",
        staging / "cpr1",
        dirs_exist_ok=True,
        copy_function=shutil.copyfile,
    )
    shutil.copytree(
        HERE / "runtime",
        staging / "runtime",
        dirs_exist_ok=True,
        copy_function=shutil.copyfile,
    )
    atomic_copy(HERE / "inflate.py", staging / "inflate.py")
    atomic_copy(HERE / "inflate.sh", staging / "inflate.sh")
    atomic_copy(vendor_file(fx2_source), staging / "runtime" / "fx2_model_axis_corrector.py")
    atomic_copy(vendor_file(residual_source), staging / "runtime" / "residual_archive.py")
    atomic_copy(archive, staging / "archive.zip")
    verify_staged_runtime(staging, archive, fx2_source, residual_source)
    if not destination.exists():
        os.replace(staging, destination)
    return destination


def compile_decoder(runtime_root: Path, build: Path) -> Path:
    library = build / "rc64_decoder.so"
    source = runtime_root / "runtime" / "entropy" / "rc64_backend.c"
    build.mkdir(parents=True, exist_ok=True)
    temporary = library.with_suffix(library.suffix + ".partial")
    command = [
        compiler_path(),
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, library)
    return library


def decode_tokens(runtime_root: Path, destination: Path, receipt_path: Path) -> dict[str, Any]:
    """Decode and retain the real 600-plane target from the pinned base archive."""
    if destination.is_file():
        fact = require_pin(destination, TOKENS_SHA256, TOKENS_BYTES)
        return {"status": "REUSED_PINNED", "tokens": fact}

    import torch

    started = time.monotonic()
    library = compile_decoder(runtime_root, destination.parent / "decoder_build")
    sys.path.insert(0, str(runtime_root))
    sys.path.insert(0, str(runtime_root / "cpr1"))
    try:
        residual = __import__("runtime.residual_archive", fromlist=["*"])
        spec = importlib.util.spec_from_file_location(
            "semantic_joint_ctxmix_compress_renderer", runtime_root / "cpr1" / "inflate.py"
        )
        if spec is None or spec.loader is None:
            raise CompressionError("cannot import the vendored CPR1 renderer")
        renderer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = renderer
        spec.loader.exec_module(renderer)
        parts = residual.read_residual_archive(runtime_root / "archive.zip")
        old_rc64 = os.environ.get("CPR1_RC64_LIBRARY")
        old_native = os.environ.pop("F26_CORRECTOR_NATIVE_LIBRARY", None)
        os.environ["CPR1_RC64_LIBRARY"] = str(library)
        try:
            tokens, decode_report = residual.decode_production_tokens(
                parts, renderer, runtime_root / "cpr1", torch.device("cpu")
            )
        finally:
            if old_rc64 is None:
                os.environ.pop("CPR1_RC64_LIBRARY", None)
            else:
                os.environ["CPR1_RC64_LIBRARY"] = old_rc64
            if old_native is not None:
                os.environ["F26_CORRECTOR_NATIVE_LIBRARY"] = old_native
        payload = tokens.numpy().tobytes(order="C")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        sys.path = [entry for entry in sys.path if entry not in {str(runtime_root), str(runtime_root / "cpr1")}]

    fact = require_pin(destination, TOKENS_SHA256, TOKENS_BYTES)
    receipt = {
        "status": "PASS",
        "axis": "[macOS-CPU advisory / scorer-free exact byte measurement]",
        "elapsed_seconds": time.monotonic() - started,
        "runtime_archive": file_fact(runtime_root / "archive.zip"),
        "decoder_library": file_fact(library),
        "tokens": fact,
        "decode_report": decode_report,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def run_jg2(
    *,
    stage: str,
    store: Path,
    runtime: Path,
    pointer: Path,
    pointer_sha256: str,
    tokens: Path,
    tag: str,
    resume: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(vendor_file("jg2_tail_reencode.py")),
        "--stage",
        stage,
        "--store",
        str(store),
        "--runtime-root",
        str(runtime),
        "--pointer-archive",
        str(pointer),
        "--expect-pointer-sha256",
        pointer_sha256,
        "--tokens",
        str(tokens),
        "--frames",
        "600",
        "--checkpoint-every",
        "25",
    ]
    if stage == "encode":
        command.extend(("--tag", tag))
    if resume:
        command.append("--resume")
    started = time.monotonic()
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise CompressionError(f"{stage} for {tag} failed with rc={completed.returncode}")
    return {"argv": command, "returncode": 0, "elapsed_seconds": time.monotonic() - started}


def encode_stage(
    *,
    label: str,
    store: Path,
    control_runtime: Path,
    encode_runtime: Path,
    pointer: Path,
    pointer_sha256: str,
    tokens: Path,
    tag: str,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    expected_sha, expected_bytes = STAGE_PINS[label]
    candidate = store / "retained" / f"candidate_{tag}.zip"
    if candidate.is_file():
        fact = require_pin(candidate, expected_sha, expected_bytes)
        return candidate, {"stage": label, "status": "REUSED_PINNED", "archive": fact}
    # The fresh public proof is gated by the exact receipt-pinned archive bytes at
    # every stage.  Re-encoding the incoming stream under ``control_runtime`` would
    # duplicate every full-n600 encode yet prove less than the following absolute
    # SHA-256 equality.  Preserve the predecessor runtime identity in the receipt
    # so the omitted differential control is explicit rather than silently absent.
    control = {
        "status": "NOT_RERUN_ABSOLUTE_OUTPUT_PIN_IS_AUTHORITY",
        "runtime_archive": file_fact(control_runtime / "archive.zip"),
    }
    encode = run_jg2(
        stage="encode",
        store=store,
        runtime=encode_runtime,
        pointer=pointer,
        pointer_sha256=pointer_sha256,
        tokens=tokens,
        tag=tag,
        resume=resume,
    )
    fact = require_pin(candidate, expected_sha, expected_bytes)
    return candidate, {
        "stage": label,
        "status": "PASS",
        "input": file_fact(pointer),
        "archive": fact,
        "control": control,
        "encode": encode,
    }


def fold_dx2_stage(
    *, store: Path, runtime: Path, pointer: Path, pointer_sha256: str
) -> tuple[Path, dict[str, Any]]:
    """Apply the real DX2 coefficient rider to the FX5 carrier body.

    DX2 is not a second token-stream encode. It replaces CAP1's Rice-coded
    600x12 coefficient payload with the measured integer CABAC representation,
    marks reserved bit 0x10, and proves that the receiver restores the original
    carrier bytes. The exact incumbent container is Brotli q9/lgwin16 with CK2
    off; rebuilding the incoming archive byte-identically is the precondition.
    """
    import brotli
    import numpy as np

    expected_sha, expected_bytes = STAGE_PINS["dx2"]
    retained = store / "retained"
    candidate_path = retained / "candidate_dx2_cabac.zip"
    result_path = retained / "RESULT.json"
    if candidate_path.is_file() and result_path.is_file():
        fact = require_pin(candidate_path, expected_sha, expected_bytes)
        previous = json.loads(result_path.read_text())
        if previous.get("status") != "PASS":
            raise CompressionError("DX2 resume receipt is not PASS")
        for payload in previous.get("retained_payloads", []):
            require_pin(
                Path(payload["path"]),
                str(payload["sha256"]),
                int(payload["bytes"]),
            )
        return candidate_path, {
            "stage": "dx2",
            "status": "REUSED_PINNED",
            "archive": fact,
        }

    require_pin(pointer, pointer_sha256, STAGE_PINS["fx5"][1])
    retained.mkdir(parents=True, exist_ok=True)

    old_path = list(sys.path)
    for stale in [name for name in sys.modules if name == "runtime" or name.startswith("runtime.")]:
        del sys.modules[stale]
    sys.path.insert(0, str(runtime))
    try:
        residual = __import__("runtime.residual_archive", fromlist=["*"])
        dx2 = __import__("runtime.dx2_cabac_coefficients", fromlist=["*"])
    finally:
        sys.path[:] = old_path
    if Path(residual.__file__).resolve() != (runtime / "runtime/residual_archive.py").resolve():
        raise CompressionError("DX2 imported a residual receiver outside its staged runtime")
    if Path(dx2.__file__).resolve() != (runtime / "runtime/dx2_cabac_coefficients.py").resolve():
        raise CompressionError("DX2 imported a coder outside its staged runtime")

    with zipfile.ZipFile(pointer) as archive:
        if archive.namelist() != ["p"]:
            raise CompressionError("DX2 input archive must contain exactly member p")
        info = archive.getinfo("p")
        outer = archive.read("p")
    header = residual.RX1_MODEL_HEADER
    if len(outer) < header.size or not outer.startswith(residual.RX1_MAGIC):
        raise CompressionError("DX2 input is not the pinned RX1M container")
    (
        magic,
        version,
        codec,
        table_mode,
        reserved,
        hpac_bytes,
        semantic_bytes,
        carrier_bytes,
    ) = header.unpack_from(outer)
    if codec != residual.RX1_CODEC_BROTLI:
        raise CompressionError("DX2 requires the Brotli RX1M codec")
    offset = header.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic_stream = outer[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    section_tail = outer[offset:]
    carrier_body = residual._decompress_brotli(carrier_stream)
    ck2 = bool(reserved & residual.CK2_RESERVED_CARRIER_PLANE2)
    if ck2:
        carrier_body = residual._ck2_uninterleave_planes(carrier_body)

    def emit(body: bytes, reserved_extra: int) -> bytes:
        staged = body
        if ck2:
            span = len(body) & ~1
            planes = np.frombuffer(body[:span], dtype=np.uint8)
            staged = planes[0::2].tobytes() + planes[1::2].tobytes() + body[span:]
        stream = brotli.compress(staged, quality=9, lgwin=16)
        if brotli.decompress(stream) != staged or len(stream) > 0xFFFF:
            raise CompressionError("DX2 incumbent Brotli container failed exact round-trip")
        output_reserved = reserved | reserved_extra
        if ck2:
            output_reserved |= residual.CK2_RESERVED_CARRIER_PLANE2
        else:
            output_reserved &= ~residual.CK2_RESERVED_CARRIER_PLANE2
        member = b"".join(
            (
                header.pack(
                    magic,
                    version,
                    codec,
                    table_mode,
                    output_reserved,
                    len(hpac_stream),
                    len(semantic_stream),
                    len(stream),
                ),
                hpac_stream,
                semantic_stream,
                stream,
                section_tail,
            )
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            entry = zipfile.ZipInfo("p", date_time=info.date_time)
            entry.compress_type = info.compress_type
            entry.external_attr = info.external_attr
            entry.create_system = info.create_system
            archive.writestr(entry, member)
        return buffer.getvalue()

    identity_archive = emit(carrier_body, 0)
    atomic_bytes(retained / "fx5_container_identity_repeat.zip", identity_archive)
    if identity_archive != pointer.read_bytes():
        raise CompressionError("DX2 cannot reproduce the incoming FX5 container byte-identically")

    applied = dx2.apply_cabac_to_carrier_body(carrier_body)
    candidate_body = bytes(applied["body"])
    cabac_payload = bytes(applied["cabac_payload"])
    rice_payload = bytes(applied["rice_payload"])
    atomic_bytes(retained / "fx5_carrier_body.bin", carrier_body)
    atomic_bytes(retained / "dx2_carrier_body.bin", candidate_body)
    atomic_bytes(retained / "dx2_rice_payload.bin", rice_payload)
    atomic_bytes(retained / "dx2_payload_adaptive_ctx_rice_cap8.bin", cabac_payload)
    for path, array in (
        (retained / "dx2_symbols.int32.npy", np.asarray(applied["symbols"], dtype=np.int32)),
        (retained / "dx2_ks.int64.npy", np.asarray(applied["ks"], dtype=np.int64)),
    ):
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    require_pin(
        retained / "dx2_payload_adaptive_ctx_rice_cap8.bin",
        DX2_PAYLOAD_SHA256,
        DX2_PAYLOAD_BYTES,
    )
    if dx2.restore_carrier_body(candidate_body) != carrier_body:
        raise CompressionError("DX2 receiver does not restore the exact FX5 carrier body")

    corrupt = bytearray(cabac_payload)
    corrupt[len(corrupt) // 2] ^= 1
    corrupt_path = retained / "negative_control_corrupt_cabac.bin"
    atomic_bytes(corrupt_path, bytes(corrupt))
    try:
        dx2.decode_cabac_checked(bytes(corrupt), applied["ks"])
    except dx2.CabacCoefficientError:
        corrupt_refused = True
    else:
        corrupt_refused = False
    if not corrupt_refused:
        raise CompressionError("DX2 corrupted-payload negative control did not fire")

    candidate = emit(candidate_body, dx2.DX2_RESERVED_CABAC_COEFFICIENTS)
    repeat = emit(candidate_body, dx2.DX2_RESERVED_CABAC_COEFFICIENTS)
    atomic_bytes(candidate_path, candidate)
    repeat_path = retained / "candidate_dx2_cabac.repeat.zip"
    atomic_bytes(repeat_path, repeat)
    fact = require_pin(candidate_path, expected_sha, expected_bytes)
    repeat_fact = require_pin(repeat_path, expected_sha, expected_bytes)
    if candidate != repeat:
        raise CompressionError("DX2 deterministic archive repeat differs")
    row = {
        "stage": "dx2",
        "status": "PASS",
        "input": file_fact(pointer),
        "archive": fact,
        "mechanism": "lossless CAP1 Rice-to-CABAC carrier fold; reserved bit 0x10",
        "controls": {
            "fx5_container_identity": file_fact(retained / "fx5_container_identity_repeat.zip"),
            "receiver_restores_carrier_body": True,
            "corrupted_cabac_refused": True,
            "deterministic_repeat": repeat_fact,
        },
        "retained_payloads": [
            file_fact(retained / name)
            for name in (
                "fx5_container_identity_repeat.zip",
                "fx5_carrier_body.bin",
                "dx2_carrier_body.bin",
                "dx2_rice_payload.bin",
                "dx2_payload_adaptive_ctx_rice_cap8.bin",
                "dx2_symbols.int32.npy",
                "dx2_ks.int64.npy",
                "negative_control_corrupt_cabac.bin",
                "candidate_dx2_cabac.repeat.zip",
            )
        ],
    }
    atomic_json(result_path, row)
    return candidate_path, row


def rebuild_once(run_number: int, store: Path, base: Path, tokens: Path, resume: bool) -> dict[str, Any]:
    started = time.monotonic()
    run_root = store / "work" / f"run_{run_number}"
    retained = store / "retained" / f"run_{run_number}"
    run_root.mkdir(parents=True, exist_ok=True)
    retained.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []

    rc2_runtime = stage_runtime(run_root / "runtime_rc2", base, "fx2_rc2.py", "residual_pre_dx2.py")
    fx5_runtime = stage_runtime(run_root / "runtime_fx5", base, "fx2_fx5.py", "residual_pre_dx2.py")
    fx5, row = encode_stage(
        label="fx5", store=run_root / "fx5", control_runtime=rc2_runtime,
        encode_runtime=fx5_runtime, pointer=base, pointer_sha256=BASE_SHA256,
        tokens=tokens, tag="fx5_e1_19member", resume=resume,
    )
    stages.append(row)
    atomic_copy(fx5, retained / "01_fx5.zip")
    atomic_copy(fx5, fx5_runtime / "archive.zip")

    dx2, row = fold_dx2_stage(
        store=run_root / "dx2",
        runtime=fx5_runtime,
        pointer=fx5,
        pointer_sha256=STAGE_PINS["fx5"][0],
    )
    stages.append(row)
    atomic_copy(dx2, retained / "02_dx2.zip")
    dx2_runtime = stage_runtime(run_root / "runtime_dx2", dx2, "fx2_fx5.py", "residual_dx2.py")
    atomic_copy(dx2, dx2_runtime / "archive.zip")

    gb1_pointer_runtime = stage_runtime(
        run_root / "runtime_gb1_pointer", dx2, "fx2_gb1_pointer.py", "residual_dx2.py"
    )
    gb1_joint_runtime = stage_runtime(
        run_root / "runtime_gb1_joint", dx2, "fx2_gb1_joint.py", "residual_dx2.py"
    )
    gb1_store = run_root / "gb1"
    gb1_pointer, row = encode_stage(
        label="gb1_pointer", store=gb1_store / "pointer", control_runtime=dx2_runtime,
        encode_runtime=gb1_pointer_runtime, pointer=dx2, pointer_sha256=STAGE_PINS["dx2"][0],
        tokens=tokens, tag="gb1_groupbin8_surprise", resume=resume,
    )
    stages.append(row)
    gb1_joint, row = encode_stage(
        label="gb1_joint", store=gb1_store / "joint", control_runtime=dx2_runtime,
        encode_runtime=gb1_joint_runtime, pointer=dx2, pointer_sha256=STAGE_PINS["dx2"][0],
        tokens=tokens, tag="gb1_joint21", resume=resume,
    )
    stages.append(row)
    atomic_copy(gb1_pointer, retained / "03a_gb1_pointer.zip")
    atomic_copy(gb1_joint, retained / "03b_gb1_joint.zip")
    atomic_copy(gb1_joint, gb1_joint_runtime / "archive.zip")

    lb1_runtime = stage_runtime(run_root / "runtime_lb1", gb1_joint, "fx2_lb1.py", "residual_dx2.py")
    lb1, row = encode_stage(
        label="lb1", store=run_root / "lb1", control_runtime=gb1_joint_runtime,
        encode_runtime=lb1_runtime, pointer=gb1_joint,
        pointer_sha256=STAGE_PINS["gb1_joint"][0], tokens=tokens,
        tag="lb1_joint22_patch192", resume=resume,
    )
    stages.append(row)
    atomic_copy(lb1, retained / "04_lb1.zip")
    atomic_copy(lb1, lb1_runtime / "archive.zip")

    afr1_runtime = stage_runtime(run_root / "runtime_afr1", lb1, "fx2_afr1.py", "residual_dx2.py")
    afr1, row = encode_stage(
        label="afr1", store=run_root / "afr1", control_runtime=lb1_runtime,
        encode_runtime=afr1_runtime, pointer=lb1, pointer_sha256=STAGE_PINS["lb1"][0],
        tokens=tokens, tag="afr1_tile48_groupbin8", resume=resume,
    )
    stages.append(row)
    final = retained / "05_afr1.zip"
    atomic_copy(afr1, final)
    fact = require_pin(final, FINAL_SHA256, FINAL_BYTES)
    result = {
        "run": run_number,
        "status": "PASS",
        "elapsed_seconds": time.monotonic() - started,
        "stages": stages,
        "final_archive": fact,
    }
    atomic_json(retained / "RESULT.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-archive", default=str(HERE / "base_archive.zip"),
        help="generation-6 rc2 archive; default is base_archive.zip beside this script",
    )
    parser.add_argument(
        "--store", type=Path, required=True,
        help="durable SSD work, checkpoints, intermediate archives, and receipts",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    global _ACTIVE_VENDOR
    args = parse_args()
    if args.repeats not in (1, 2):
        raise CompressionError(
            "--repeats must be 1 (SHA-pinned rebuild) or 2 (adds a full re-run to demonstrate determinism)"
        )
    args.store = args.store.expanduser().resolve()
    if args.store == HERE or HERE in args.store.parents:
        raise CompressionError("--store must be outside the public source tree")
    args.store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.store).free
    receipt: dict[str, Any] = {
        "schema": "semantic_joint_ctxmix.compress.v1",
        "status": "RUNNING",
        "axis": "[macOS-CPU advisory / scorer-free exact byte measurement]",
        "score_claim": False,
        "started_unix": time.time(),
        "storage_preflight": {"free_bytes": free, "minimum_bytes": 8 << 30},
        "runs": [],
    }
    receipt_path = args.store / "RESULT.json"
    atomic_json(receipt_path, receipt)
    try:
        if free < 8 << 30:
            raise CompressionError(f"storage preflight requires 8 GiB free; found {free} bytes")
        tree_rows = verify_tree_manifest()
        receipt["environment"] = dependency_fingerprint()
        _ACTIVE_VENDOR = materialize_encoder_sources(args.store)
        receipt["source_pins"] = verify_sources(tree_rows)
        atomic_json(receipt_path, receipt)
        base = obtain_base(args)
        receipt["base_archive"] = require_pin(base, BASE_SHA256, BASE_BYTES)
        decode_runtime = stage_runtime(
            args.store / "work" / "decode_runtime",
            base,
            "fx2_rc2.py",
            "residual_pre_dx2.py",
        )
        tokens = args.store / "retained" / "inputs" / "tokens.u8"
        receipt["token_decode"] = decode_tokens(
            decode_runtime, tokens, args.store / "retained" / "TOKEN_DECODE.json"
        )
        atomic_json(receipt_path, receipt)
        for run_number in range(1, args.repeats + 1):
            result = rebuild_once(run_number, args.store, base, tokens, args.resume)
            receipt["runs"].append(result)
            atomic_json(receipt_path, receipt)
        first = Path(receipt["runs"][0]["final_archive"]["path"])
        if args.repeats == 2:
            second = Path(receipt["runs"][1]["final_archive"]["path"])
            if first.read_bytes() != second.read_bytes():
                raise CompressionError("the two complete rebuilds differ byte-for-byte")
            receipt["determinism"] = {
                "mode": "two_full_rebuilds",
                "byte_identical": True,
                "run_1": file_fact(first),
                "run_2": file_fact(second),
            }
        else:
            receipt["determinism"] = {
                "mode": "single_rebuild_sha_pinned",
                "note": (
                    "correctness is enforced by the FINAL_SHA256 pin below and by the "
                    "per-stage in-memory determinism repeats; pass --repeats 2 for a "
                    "full second-rebuild demonstration"
                ),
                "run_1": file_fact(first),
            }
        delivered = args.store / "retained" / "archive.zip"
        atomic_copy(first, delivered)
        receipt["archive"] = require_pin(delivered, FINAL_SHA256, FINAL_BYTES)
        receipt["status"] = "PASS"
        receipt["elapsed_seconds"] = time.time() - receipt["started_unix"]
        atomic_json(receipt_path, receipt)
    except Exception as error:
        receipt["status"] = "REFUSED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        atomic_json(receipt_path, receipt)
        raise
    print(json.dumps({"status": "PASS", "archive": receipt["archive"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
