import sys, numpy as np, open3d as o3d, copy
sys.path.insert(0,'/mnt/project-files/code/vgap')
import vgap_gate as vg
np.random.seed(0); o3d.utility.random.seed(0)
K=np.array([[609.3,0,326.1],[0,608.9,249.2],[0,0,1]]); H,W=480,640
mesh=o3d.io.read_triangle_mesh('/mnt/project-files/cad/gearbox/gearbox_assembled_100mm_in_meters.stl')
mesh.compute_triangle_normals()
V=np.asarray(mesh.vertices); F=np.asarray(mesh.triangles)
tri_c=V[F].mean(1); is_lip=np.abs(tri_c[:,0])>0.0729/2+1e-4   # outside body in x
# exterior CAD cloud (like their inside-removed pcd): sample, keep points visible from outside
cad_dense=o3d.io.read_point_cloud('/mnt/project-files/cad/gearbox/gearbox_exterior_100mm_m.ply')
cad=cad_dense.voxel_down_sample(0.0017)
# orient CAD normals outward from centroid (box-like)


def look_pose(tilt_deg, az_deg, yaw_deg, dist=0.40):
    # object at origin on table (z up, bottom at z=0 -> shift), camera on sphere
    t,a=np.radians(tilt_deg),np.radians(az_deg)
    cpos=dist*np.array([np.sin(t)*np.cos(a),np.sin(t)*np.sin(a),np.cos(t)])
    z=-cpos/np.linalg.norm(cpos); up=np.array([0,1.,0]) if tilt_deg<1 else np.array([0,0,1.])
    x=np.cross(z,up) if tilt_deg>=1 else np.array([1.,0,0]); x/=np.linalg.norm(x); y=np.cross(z,x)
    R_wc=np.stack([x,y,z],1)  # cam axes in world
    T_wc=np.eye(4); T_wc[:3,:3]=R_wc; T_wc[:3,3]=cpos
    T_ow=np.eye(4); c,s=np.cos(np.radians(yaw_deg)),np.sin(np.radians(yaw_deg)); T_ow[:2,:2]=[[c,-s],[s,c]]
    return np.linalg.inv(T_wc)@T_ow   # object->camera

def render(T):
    scene=o3d.t.geometry.RaycastingScene()
    m=copy.deepcopy(mesh); m.transform(T)
    tm=o3d.t.geometry.TriangleMesh.from_legacy(m); scene.add_triangles(tm)
    # table plane below object: skip (mask crop removes it anyway)
    rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(o3d.core.Tensor(K),o3d.core.Tensor(np.eye(4)),W,H)
    ans=scene.cast_rays(rays)
    t=ans['t_hit'].numpy(); prim=ans['primitive_ids'].numpy()
    hit=np.isfinite(t)
    d=rays.numpy()[...,3:]; zc=np.where(hit,t*d[...,2],0)
    depth=(zc*1000).astype(np.uint16)
    lip=np.zeros_like(hit); lip[hit]=is_lip[prim[hit]]
    return depth,hit,lip

def cloud(depth):
    img=o3d.geometry.Image(depth); intr=o3d.camera.PinholeCameraIntrinsic(W,H,K[0,0],K[1,1],K[0,2],K[1,2])
    pc=o3d.geometry.PointCloud.create_from_depth_image(img,intr,depth_scale=1000.0,depth_trunc=0.8)
    pc=pc.voxel_down_sample(0.0017)
    pc.points=o3d.utility.Vector3dVector(np.asarray(pc.points)+np.random.normal(0,0.0007,(len(pc.points),3)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.0034*1.5,max_nn=30)); pc.orient_normals_towards_camera_location(np.zeros(3))
    return pc

def register(src):
    v=0.0017
    f=lambda pc: o3d.pipelines.registration.compute_fpfh_feature(pc,o3d.geometry.KDTreeSearchParamHybrid(radius=v*5,max_nn=100))
    tvec=src.get_center()-cad.get_center(); Ti=np.eye(4); Ti[:3,3]=tvec
    tgt=copy.deepcopy(cad); tgt.translate(tvec)
    r=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(src,tgt,f(src),f(tgt),True,v*1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.88),o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(v*1.5),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.cos(np.radians(45)))],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000,1000))
    T=r.transformation
    for th in [0.012*2/3,0.010*2/3,0.008*2/3]:
        T=o3d.pipelines.registration.registration_icp(src,tgt,th,T,o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)).transformation
    return np.linalg.inv(T)@Ti

G=vg.gearbox_symmetry_set(np.asarray(cad.points))
dense=np.asarray(cad_dense.points)
def err(Tg,Te): return vg.symmetric_rotation_error_deg(Tg[:3,:3],Te[:3,:3],G), np.linalg.norm(Tg[:3,3]-Te[:3,3])*1000
rows=[]
for tilt in [0,30,45]:
  for yaw in [0,35]:
    Tgt=look_pose(tilt,0,yaw)
    depth,M2D,lip=render(Tgt)
    ys,xs=np.nonzero(M2D); B2D=[xs.min(),ys.min(),xs.max(),ys.max()]
    for lips in ['with lips','lips dropped']:
      d=depth.copy()
      if lips=='lips dropped': d[lip]=0
      d=vg.masked_depth(d,M2D,erode_px=1)
      src=cloud(d)
      # A) baseline registration, then gate + yaw resolution
      for trial in range(3):
        Te=register(src)
        e0=err(Tgt,Te)
        Mp,Bp=vg.project_silhouette(dense,Te,K,(H,W)); C0=vg.consistency(M2D,B2D,Mp,Bp)[2]
        r=vg.resolve_yaw(src,cad,Te,K,M2D,B2D,(H,W),dense)
        e1=err(Tgt,r['T'])
        rows.append((tilt,yaw,lips,trial,round(e0[0],1),round(C0,2),round(e1[0],1),round(r['C'],2),round(r['margin'],2)))
        print(*rows[-1],flush=True)
      # B) injected 90 deg yaw error
      c,ax=vg.principal_frame(np.asarray(cad.points))
      Tbad=Tgt@vg.rotation_about(c,ax[:,2],90)
      Mp,Bp=vg.project_silhouette(dense,Tbad,K,(H,W)); Cb=vg.consistency(M2D,B2D,Mp,Bp)[2]
      r=vg.resolve_yaw(src,cad,Tbad,K,M2D,B2D,(H,W),dense)
      print('  injected90',tilt,yaw,lips,'C_bad',round(Cb,2),'-> after resolve err',round(err(Tgt,r['T'])[0],1),'C',round(r['C'],2),'margin',round(r['margin'],2),flush=True)
