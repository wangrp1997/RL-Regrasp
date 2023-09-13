# Copyright (c) 2016, Universal Robots A/S,
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Universal Robots A/S nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL UNIVERSAL ROBOTS A/S BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import struct
import socket
import select
import sys
import logging

if sys.version_info[0] < 3:
  from ur_control import serialize
else:
  from ur_control import serialize

DEFAULT_TIMEOUT = 1.0

LOGNAME = 'rtde'
_log = logging.getLogger(LOGNAME)


class Command:
    RTDE_REQUEST_PROTOCOL_VERSION = 86        # ascii V
    RTDE_GET_URCONTROL_VERSION = 118          # ascii v
    RTDE_TEXT_MESSAGE = 77                    # ascii M
    RTDE_DATA_PACKAGE = 85                    # ascii U
    RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS = 79   # ascii O
    RTDE_CONTROL_PACKAGE_SETUP_INPUTS = 73    # ascii I
    RTDE_CONTROL_PACKAGE_START = 83           # ascii S
    RTDE_CONTROL_PACKAGE_PAUSE = 80           # ascii P


RTDE_PROTOCOL_VERSION = 2

class ConnectionState:
    DISCONNECTED = 0
    CONNECTED = 1
    STARTED = 2
    PAUSED = 3

class RTDEException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

class RTDE(object):
    def __init__(self, hostname, port=30004):
        self.hostname = hostname
        self.port = port
        self.__conn_state = ConnectionState.DISCONNECTED
        self.__sock = None
        self.__output_config = None
        self.__input_config = {}

    def connect(self):
        if self.__sock:
            return

        self.__buf = b'' # buffer data in binary format
        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.__sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.__sock.settimeout(DEFAULT_TIMEOUT)
            self.__sock.connect((self.hostname, self.port))
            self.__conn_state = ConnectionState.CONNECTED
        except (socket.timeout, socket.error):
            self.__sock = None
            raise
        if not self.negotiate_protocol_version():
            raise RTDEException('Unable to negotiate protocol version')

    def disconnect(self):
        if self.__sock:
            self.__sock.close()
            self.__sock = None
        self.__conn_state = ConnectionState.DISCONNECTED

    def is_connected(self):
        return self.__conn_state is not ConnectionState.DISCONNECTED

    def get_controller_version(self):
        cmd = Command.RTDE_GET_URCONTROL_VERSION
        version = self.__sendAndReceive(cmd)
        if version:
            _log.info('Controller version: ' + str(version.major) + '.' + str(version.minor) + '.' + str(version.bugfix)+ '.' + str(version.build))
            if version.major == 3 and version.minor <= 2 and version.bugfix < 19171:
                _log.error("Please upgrade your controller to minimally version 3.2.19171")
                sys.exit()
            return version.major, version.minor, version.bugfix, version.build
        return None, None, None, None

    def negotiate_protocol_version(self):
        cmd = Command.RTDE_REQUEST_PROTOCOL_VERSION
        payload = struct.pack('>H', RTDE_PROTOCOL_VERSION)
        success = self.__sendAndReceive(cmd, payload)
        return success

    def send_input_setup(self, variables, types=[]):
        cmd = Command.RTDE_CONTROL_PACKAGE_SETUP_INPUTS
        payload = bytearray(','.join(variables), 'utf-8')
        result = self.__sendAndReceive(cmd, payload)
        if len(types)!=0 and not self.__list_equals(result.types, types):
            _log.error('Data type inconsistency for input setup: ' +
                     str(types) + ' - ' +
                     str(result.types))
            return None
        result.names = variables
        self.__input_config[result.id] = result
        return serialize.DataObject.create_empty(variables, result.id)

    def send_output_setup(self, variables, types=[], frequency=125):
        cmd = Command.RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS
        payload = struct.pack('>d', frequency)
        payload = payload + (','.join(variables).encode('utf-8'))
        result = self.__sendAndReceive(cmd, payload)
        if len(types)!=0 and not self.__list_equals(result.types, types):
            _log.error('Data type inconsistency for output setup: ' +
                     str(types) + ' - ' +
                     str(result.types))
            return False
        result.names = variables
        self.__output_config = result
        return True

    def send_start(self):
        cmd = Command.RTDE_CONTROL_PACKAGE_START
        success = self.__sendAndReceive(cmd)
        if success:
            _log.info('RTDE synchronization started')
            self.__conn_state = ConnectionState.STARTED
        else:
            _log.error('RTDE synchronization failed to start')
        return success

    def send_pause(self):
        cmd = Command.RTDE_CONTROL_PACKAGE_PAUSE
        success = self.__sendAndReceive(cmd)
        if success:
            _log.info('RTDE synchronization paused')
            self.__conn_state = ConnectionState.PAUSED
        else:
            _log.error('RTDE synchronization failed to pause')
        return success

    def send(self, input_data):
        if self.__conn_state != ConnectionState.STARTED:
            _log.error('Cannot send when RTDE synchronization is inactive')
            return
        if not input_data.recipe_id in self.__input_config:
            _log.error('Input configuration id not found: ' + str(input_data.recipe_id))
            return
        config = self.__input_config[input_data.recipe_id]
        return self.__sendall(Command.RTDE_DATA_PACKAGE, config.pack(input_data))

    def receive(self, binary=False):
        if self.__output_config is None:
            _log.error('Output configuration not initialized')
            return None
        if self.__conn_state != ConnectionState.STARTED:
            _log.error('Cannot receive when RTDE synchronization is inactive')
            return None
        return self.__recv(Command.RTDE_DATA_PACKAGE, binary)

    def send_message(self, message, source = "Python Client", type = serialize.Message.INFO_MESSAGE):
        cmd = Command.RTDE_TEXT_MESSAGE
        fmt = '>B%dsB%dsB' % (len(message), len(source))
        payload = struct.pack(fmt, len(message), message, len(source), source, type)
        return self.__sendall(cmd, payload)

    def __on_packet(self, cmd, payload):
        if cmd == Command.RTDE_REQUEST_PROTOCOL_VERSION:
            return self.__unpack_protocol_version_package(payload)
        elif cmd == Command.RTDE_GET_URCONTROL_VERSION:
            return self.__unpack_urcontrol_version_package(payload)
        elif cmd == Command.RTDE_TEXT_MESSAGE:
            return self.__unpack_text_message(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS:
            return self.__unpack_setup_outputs_package(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_SETUP_INPUTS:
            return self.__unpack_setup_inputs_package(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_START:
            return self.__unpack_start_package(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_PAUSE:
            return self.__unpack_pause_package(payload)
        elif cmd == Command.RTDE_DATA_PACKAGE:
            return self.__unpack_data_package(payload, self.__output_config)
        else:
            _log.error('Unknown package command: ' + str(cmd))

    def __sendAndReceive(self, cmd, payload=b''):
        if self.__sendall(cmd, payload):
            return self.__recv(cmd)
        else:
            return None

    def __sendall(self, command, payload=b''):
        fmt = '>HB'
        size = struct.calcsize(fmt) + len(payload)
        buf = strtct&pack(fmt,�uije,){ommanl)"� `aylgad

        if&se|f.__Soc+ i{ �one:
$    0      _lo�.arro�('�mable to cend:(n�l connecxd qo Robot'+
      `�    rettrn FalseJ
 $     ��,!�ri|eble { =`select.relect,KU�`[welf__qocO],0[], DEFAULU[\IMEO]T) ( 0  ` i& len�v�itabl!):
  0  0      self/__cokk.sendall,cuvi
        * ` retwRo$Tr�eJ0� $ (  alse:
�      !    welf/O_trigger_eh{connec�ed()(      !    vettrn V�lqe

 $ �deb har_data�semf);
(d  �   tymeoqt 9 2
�`  �0  reedaRle, _, _8= qelact:canekt([sglf.__sock], []. [],`t(mEout)
  &`    Bedwrn len(�ea�able)!=0
*  " `ef _[beavhelf, gomoand$ bi.a�yNamrD):
(       whal%1welf�is_#on�ectel(+;
 (    $  d  bmafable,r_, xljsT0= sele;t*sule�t([sEmf*N�sockY, []l Zqg,g.]�{OckX, DEFALT_TI]EMUT)* * % 0  �0 0�f l�n(RecDable#8�  �   4 !  � 0 `more 5 self._Wqocc.rec(4�2)ʢ     t  $      i& le�(}ove) =="0:
(     � `     `     self._4ryG�cr_dqc#on.acted(-
       "      `  0 return None
    �()         s%hf.__buf�= wEnf.__bu& � =ore

(          ig ldnx,ist) oz8len(fdadable) == 0:�# M�fmstivelz@A lImeout _& DEFAQMp_tIEO\T secnndw     ` $   �  �_log"inf('lort colnec4io.$vi�h$cojtrllerg)
           `   @se.f._�trigge~Wdisconnectee()
  h      �  �"  reTtvl None
             wnpack�from reQuirEs a buffur"oF�at least 2 byte{ �   >0     Whi�u�len(3e~f._\buf)$>= 2:  �  (   ``#    # Attempts to e|tr!bt!a 0akkev
     ` �   "4   p!cket_heal�r =)rmriatize.ControlLe`dar*unpack*self.__bUf)
� !(&     $    if len8saL�.__rqF�!?= racKuT_header.cize:``(�!0   $ ! �    � packet, sqlf,__bwf y self.__buf[38packetXhe�der.wIze],(s%lf._^buF[pecomt_head%z&saze:]      !  "  !      $dapa = sel&.W_ol_xacket8pac{eaWhEader.cgmmand, pab�et)  !3         �  $   if ,ensglf.__fU�)d>= : a.$ C/|mA�dj== Bkemand.RMDeDCTAOPAC�AGE:  $ !        8      !$  �uxt_01ske|Oh�der = {�viqliZl.Konvsml��ader.u~pa#k(s%lFn__bq��
            0"  !       yn naxt_qccket_he�dev.co�lan� =9 comman$:J( (      ( "    !`  " " (   _l/'.in&o('skiDpile$paAkage(1!')
    ""   �    �(    0h  $   continue
            !       yf!p!G�et_�ea�ep.cOmmand8=�$cOmman`:
  !   (�    � �!    $"( if)bknarY):
`" 2    �       �b      0$ $rmt}fm packet[13]
J  (b �   �       @ 0    rdt�rn data
�    H$ $(  0     " eese:  `           $   �"    _nog4`nfo('skipping xackoGd(2)'9
      0( `  0 ep�e;
 $         �  ($� 8 breAk ""     return NOno

    dg� _tpiOger_dic#onnected(self): 0 `0h  _�oo.info(RTDE dys�nllmcted*+
  %� ` welb,dicc$n.egt() "clean-�p
J   eef __wnpack_prkToco|]v%rrion_0ackage�self, paylo!d)>
!"    # if l�lrsy|fAd) != 1:
` (  �      ]log.ertos(gRTDG^REQ�ESV_PROTOGOL_VERS	ON: Wronf piyhka4 shze7)�$   `"  ( "!rEtujn nne
$(&  �  2es�l� = sgriali:e.REt�rn^alem.npaco(0ayloau)
 `     rutu�n result/su#cEss

 b0 ief _WunpcCk_urcontr_$[vgrsimn_paciage(s%,f,0Pay|!d�:�        �f$�en(deyloed) != 16:*80        ( _log.error(�RTdE_OETEUSCoNP_L_VERSI�N: gvong payloct size')    @�`  -  re�urg None
�   $   version - smriali~e.Controlersin.unpack(pa�loa$)        be�urn0versikn    def _unpa#k]ve�t_m!sw`ge*self, payl�afi:J `   ,  i� leN*payload) $ 1
            [hog.err>r('[TDE�TEXT_MERSCGE: Nf rayloqd'+        $! �et=rn Nohe
   0d0` Msu$="sesialixe.M�ss!am.tnpeck(`eyload)
  � ( 8 )f(-sg,l%ve� == Sqri�$m:e.Ma�saoe.EXAPTIONWMESSHOG$�r
� (        m2g*ngvel = sezialize,]grs�ge.E�ROR�MMSSAGEh�
          ( _loc,e2RoR(i;g.sourcE + ':  k msc.mESuegE)
        mlif0-sg.level == sebial�zg"Gessage.WA&ING_�ES�AGE��   0    ` � _log.7croing�ms�&3.srcd`+ �:$& + m3g.mgscage9
 (      emif og*,evel =="serif|yze}essaenK^FGWMESSAGE2
   `     � "_log.info(M3g.so5rcD$+!'z ' + mrg�eescq'e)
    $af _^uopakc_setup_ou�qetspackage({elf, paqloag): (!     if nen(payl~af- < 1: � "     (! _.og.esror(7BTDE_CONvQGL_PACKAGE]SE\UPOQTPU\S: Nm�payLoad')�  ! * "     re�|vn0N�Ne  `     output_condio =$serianiz�.DaTAonnig.unpaco]Re3iPa(paqoad)
 !   ( ,re�wr6 oudput_con�cfJ
(   vef __u&pack_sep~p[inputW_pcckagU(self$ payLmAd):B  (   @ yf L�n*qayloa$) <"1:
  $       $ _lk'.e�ro28'RTD�_CONRGL�PACKAGE�WETUP_MNPUTS No XaYnoi$')
$  �    ! ��repurn F+ndN     `  mFputconfi'`} sar{alije.EataBonvig.unpa#gWrecihe*pA}hoat)
 `#  $ReTuro InpUpcnnf+g�    eef __u~pa#ksdart�pac�aGe(qelg,$`e{locd :
     �  id lev(pa�lom�) %�(1:
  @        %_l/g.error('RTDE_CONDP�LWP!C�AE_QTARTz"Vrong p`{load Size')
  "         v%tu~n!No.e
 "  � " result = serIalize.RetunValqe.wFtack(pay,oaf)  ( ! ` re|u�N`sas|t.suscess

!   def0[_}npicc_xau3e_packageself, payload):
�       �f len(paylmed) != 1� ` !      " ,�G&erro�('RPDE_CL^\ROL_PacKAGE_PEUSE� Wrgng P�yloaD size/(
!�          PetuPn0Honu
       ,recult,=!ser)aliZe.ReturfWalqe.unpack(pa�load9
   � � "pdyurn resulv&stccess

    Def__ejpac+_d�t�Opd#kage(self, pe}lo�$� oe4pu6_confiw)>
`      if gutpwt_cond�� iC1None:
 $    $ "`  _l�g.esror,'RTDE[DAtA[ACKAGE:!MissilG$ou|put jonfiw�v!ti/n')
         �  2eur. Non%
�0      outut 9 oup�qt]con�ig.unpAsC(paylo`d)
(*"  `  vet52n /utpu�

    ee� __l+sv�Gquals(cel�, l1\ |2):
        ig len(l9  = len(l2):*  &   �     2aturn F�lse (      for i i�`rbnee(men((l1))):J   $       $yf!l1Si]`!= l2[y]:�   1!  0        reTurn"Fals�
 $!! "  retUrn4Tpqe