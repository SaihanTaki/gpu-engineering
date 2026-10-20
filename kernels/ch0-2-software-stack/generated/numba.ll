; ModuleID = "saxpy$1"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

@"_ZN08NumbaEnv5saxpy5saxpyB2v1B92cw51cXTLSUwv1sDUaKthoaNgoKmjgOR3W3CwAkMXLaJtQYkOIgxJU0gCqOkEJoHkbttqdVhoqlspQGNFHSgJ5BnXagIAEf5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE" = common global i8* null
define void @"_ZN5saxpy5saxpyB2v1B92cw51cXTLSUwv1sDUaKthoaNgoKmjgOR3W3CwAkMXLaJtQYkOIgxJU0gCqOkEJoHkbttqdVhoqlspQGNFHSgJ5BnXagIAEf5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE"(float %"a", i8* %"x.0", i8* %"x.1", i64 %"x.2", i64 %"x.3", float* %"x.4", i64 %"x.5.0", i64 %"x.6.0", i8* %"y.0", i8* %"y.1", i64 %"y.2", i64 %"y.3", float* %"y.4", i64 %"y.5.0", i64 %"y.6.0", i8* %"out.0", i8* %"out.1", i64 %"out.2", i64 %"out.3", float* %"out.4", i64 %"out.5.0", i64 %"out.6.0")
{
entry:
  %"inserted.meminfo" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} undef, i8* %"x.0", 0
  %"inserted.parent" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.meminfo", i8* %"x.1", 1
  %"inserted.nitems" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.parent", i64 %"x.2", 2
  %"inserted.itemsize" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.nitems", i64 %"x.3", 3
  %"inserted.data" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.itemsize", float* %"x.4", 4
  %".25" = insertvalue [1 x i64] undef, i64 %"x.5.0", 0
  %"inserted.shape" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.data", [1 x i64] %".25", 5
  %".26" = insertvalue [1 x i64] undef, i64 %"x.6.0", 0
  %"inserted.strides" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.shape", [1 x i64] %".26", 6
  %"inserted.meminfo.1" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} undef, i8* %"y.0", 0
  %"inserted.parent.1" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.meminfo.1", i8* %"y.1", 1
  %"inserted.nitems.1" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.parent.1", i64 %"y.2", 2
  %"inserted.itemsize.1" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.nitems.1", i64 %"y.3", 3
  %"inserted.data.1" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.itemsize.1", float* %"y.4", 4
  %".27" = insertvalue [1 x i64] undef, i64 %"y.5.0", 0
  %"inserted.shape.1" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.data.1", [1 x i64] %".27", 5
  %".28" = insertvalue [1 x i64] undef, i64 %"y.6.0", 0
  %"inserted.strides.1" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.shape.1", [1 x i64] %".28", 6
  %"inserted.meminfo.2" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} undef, i8* %"out.0", 0
  %"inserted.parent.2" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.meminfo.2", i8* %"out.1", 1
  %"inserted.nitems.2" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.parent.2", i64 %"out.2", 2
  %"inserted.itemsize.2" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.nitems.2", i64 %"out.3", 3
  %"inserted.data.2" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.itemsize.2", float* %"out.4", 4
  %".29" = insertvalue [1 x i64] undef, i64 %"out.5.0", 0
  %"inserted.shape.2" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.data.2", [1 x i64] %".29", 5
  %".30" = insertvalue [1 x i64] undef, i64 %"out.6.0", 0
  %"inserted.strides.2" = insertvalue {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.shape.2", [1 x i64] %".30", 6
  %"a.1" = alloca float
  store float 0.0, float* %"a.1"
  %"x" = alloca {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"x"
  %"y" = alloca {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"y"
  %"out" = alloca {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  %"i" = alloca i64
  store i64 0, i64* %"i"
  %".52" = alloca {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".52"
  %"$74pred" = alloca i1
  store i1 0, i1* %"$74pred"
  %".69" = alloca {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69"
  %".107" = alloca {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107"
  %".142" = alloca {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142"
  br label %"B0"
B0:
  store float %"a", float* %"a.1"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.strides", {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"x"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.strides.1", {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"y"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %"inserted.strides.2", {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  %".39" = trunc i64 1 to i32
  %".40" = call i32 @"llvm.nvvm.read.ptx.sreg.tid.x"()
  %".41" = sext i32 %".40" to i64
  %".42" = call i32 @"llvm.nvvm.read.ptx.sreg.ntid.x"()
  %".43" = sext i32 %".42" to i64
  %".44" = call i32 @"llvm.nvvm.read.ptx.sreg.ctaid.x"()
  %".45" = sext i32 %".44" to i64
  %".46" = mul i64 %".43", %".45"
  %".47" = add i64 %".46", %".41"
  %".49" = load i64, i64* %"i"
  store i64 %".47", i64* %"i"
  %".51" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".52"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %".51", {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".52"
  %".56" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".52", i32 0, i32 2
  %".57" = load i64, i64* %".56"
  %".58" = load i64, i64* %"i"
  %".59" = icmp slt i64 %".58", %".57"
  %".61" = load i1, i1* %"$74pred"
  store i1 %".59", i1* %"$74pred"
  %".63" = load i1, i1* %"$74pred"
  br i1 %".63", label %"B76", label %"B112"
B76:
  %".65" = load i1, i1* %"$74pred"
  store i1 0, i1* %"$74pred"
  %".67" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"x"
  %".68" = load i64, i64* %"i"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %".67", {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69"
  %".73" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69", i32 0, i32 5
  %".74" = getelementptr inbounds [1 x i64], [1 x i64]* %".73", i32 0, i32 0
  %".75" = load i64, i64* %".74", !range !1
  %".76" = insertvalue [1 x i64] undef, i64 %".75", 0
  %".77" = extractvalue [1 x i64] %".76", 0
  %".78" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69", i32 0, i32 6
  %".79" = load [1 x i64], [1 x i64]* %".78"
  %".80" = extractvalue [1 x i64] %".79", 0
  %".81" = icmp slt i64 %".68", 0
  %".82" = add i64 %".68", %".77"
  %".83" = select  i1 %".81", i64 %".82", i64 %".68"
  %".84" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69", i32 0, i32 5
  %".85" = getelementptr inbounds [1 x i64], [1 x i64]* %".84", i32 0, i32 0
  %".86" = load i64, i64* %".85", !range !1
  %".87" = insertvalue [1 x i64] undef, i64 %".86", 0
  %".88" = extractvalue [1 x i64] %".87", 0
  %".89" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69", i32 0, i32 6
  %".90" = load [1 x i64], [1 x i64]* %".89"
  %".91" = extractvalue [1 x i64] %".90", 0
  %".92" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".69", i32 0, i32 4
  %".93" = load float*, float** %".92"
  %".94" = mul i64 %".91", %".83"
  %".95" = ptrtoint float* %".93" to i64
  %".96" = add i64 %".95", %".94"
  %".97" = inttoptr i64 %".96" to float*
  %".98" = load float, float* %".97"
  %".99" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"x"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"x"
  %".101" = load float, float* %"a.1"
  %".102" = fmul float %".101", %".98"
  %".103" = load float, float* %"a.1"
  store float 0.0, float* %"a.1"
  %".105" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"y"
  %".106" = load i64, i64* %"i"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %".105", {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107"
  %".111" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107", i32 0, i32 5
  %".112" = getelementptr inbounds [1 x i64], [1 x i64]* %".111", i32 0, i32 0
  %".113" = load i64, i64* %".112", !range !1
  %".114" = insertvalue [1 x i64] undef, i64 %".113", 0
  %".115" = extractvalue [1 x i64] %".114", 0
  %".116" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107", i32 0, i32 6
  %".117" = load [1 x i64], [1 x i64]* %".116"
  %".118" = extractvalue [1 x i64] %".117", 0
  %".119" = icmp slt i64 %".106", 0
  %".120" = add i64 %".106", %".115"
  %".121" = select  i1 %".119", i64 %".120", i64 %".106"
  %".122" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107", i32 0, i32 5
  %".123" = getelementptr inbounds [1 x i64], [1 x i64]* %".122", i32 0, i32 0
  %".124" = load i64, i64* %".123", !range !1
  %".125" = insertvalue [1 x i64] undef, i64 %".124", 0
  %".126" = extractvalue [1 x i64] %".125", 0
  %".127" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107", i32 0, i32 6
  %".128" = load [1 x i64], [1 x i64]* %".127"
  %".129" = extractvalue [1 x i64] %".128", 0
  %".130" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".107", i32 0, i32 4
  %".131" = load float*, float** %".130"
  %".132" = mul i64 %".129", %".121"
  %".133" = ptrtoint float* %".131" to i64
  %".134" = add i64 %".133", %".132"
  %".135" = inttoptr i64 %".134" to float*
  %".136" = load float, float* %".135"
  %".137" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"y"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"y"
  %".139" = fadd float %".102", %".136"
  %".140" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  %".141" = load i64, i64* %"i"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} %".140", {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142"
  %".146" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142", i32 0, i32 5
  %".147" = getelementptr inbounds [1 x i64], [1 x i64]* %".146", i32 0, i32 0
  %".148" = load i64, i64* %".147", !range !1
  %".149" = insertvalue [1 x i64] undef, i64 %".148", 0
  %".150" = extractvalue [1 x i64] %".149", 0
  %".151" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142", i32 0, i32 6
  %".152" = load [1 x i64], [1 x i64]* %".151"
  %".153" = extractvalue [1 x i64] %".152", 0
  %".154" = icmp slt i64 %".141", 0
  %".155" = add i64 %".141", %".150"
  %".156" = select  i1 %".154", i64 %".155", i64 %".141"
  %".157" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142", i32 0, i32 5
  %".158" = getelementptr inbounds [1 x i64], [1 x i64]* %".157", i32 0, i32 0
  %".159" = load i64, i64* %".158", !range !1
  %".160" = insertvalue [1 x i64] undef, i64 %".159", 0
  %".161" = extractvalue [1 x i64] %".160", 0
  %".162" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142", i32 0, i32 6
  %".163" = load [1 x i64], [1 x i64]* %".162"
  %".164" = extractvalue [1 x i64] %".163", 0
  %".165" = getelementptr inbounds {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %".142", i32 0, i32 4
  %".166" = load float*, float** %".165"
  %".167" = mul i64 %".164", %".156"
  %".168" = ptrtoint float* %".166" to i64
  %".169" = add i64 %".168", %".167"
  %".170" = inttoptr i64 %".169" to float*
  store float %".139", float* %".170"
  %".172" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  %".174" = load i64, i64* %"i"
  store i64 0, i64* %"i"
  ret void
B112:
  %".178" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"y"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"y"
  %".180" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"x"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"x"
  %".182" = load {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  store {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}* %"out"
  %".184" = load i64, i64* %"i"
  store i64 0, i64* %"i"
  %".186" = load float, float* %"a.1"
  store float 0.0, float* %"a.1"
  %".188" = load i1, i1* %"$74pred"
  store i1 0, i1* %"$74pred"
  ret void
}

declare i32 @"llvm.nvvm.read.ptx.sreg.tid.x"()

declare i32 @"llvm.nvvm.read.ptx.sreg.ntid.x"()

declare i32 @"llvm.nvvm.read.ptx.sreg.ctaid.x"()

@"llvm.used" = appending global [1 x i8*] [i8* bitcast (void (float, i8*, i8*, i64, i64, float*, i64, i64, i8*, i8*, i64, i64, float*, i64, i64, i8*, i8*, i64, i64, float*, i64, i64)* @"_ZN5saxpy5saxpyB2v1B92cw51cXTLSUwv1sDUaKthoaNgoKmjgOR3W3CwAkMXLaJtQYkOIgxJU0gCqOkEJoHkbttqdVhoqlspQGNFHSgJ5BnXagIAEf5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE" to i8*)], section "llvm.metadata"
!nvvmir.version = !{ !0 }
!nvvm.annotations = !{ !2 }
!0 = !{ i32 2, i32 0, i32 3, i32 2 }
!1 = !{ i64 0, i64 9223372036854775807 }
!2 = !{ void (float, i8*, i8*, i64, i64, float*, i64, i64, i8*, i8*, i64, i64, float*, i64, i64, i8*, i8*, i64, i64, float*, i64, i64)* @"_ZN5saxpy5saxpyB2v1B92cw51cXTLSUwv1sDUaKthoaNgoKmjgOR3W3CwAkMXLaJtQYkOIgxJU0gCqOkEJoHkbttqdVhoqlspQGNFHSgJ5BnXagIAEf5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE", !"kernel", i32 1 }
